#include "common.h"
#include "common-whisper.h"
#include "whisper.h"

#include <cstdio>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>
#include <cstring>

// command-line parameters
struct whisper_params {
    int32_t n_threads  = std::min(4, (int32_t) std::thread::hardware_concurrency());
    // Transcribe the whole utterance at the end marker: shorter windows re-transcribe the kept overlap and duplicate words.
    int32_t length_ms  = 30000;
    int32_t keep_ms    = 1000;
    int32_t max_tokens = 32;
    int32_t audio_ctx  = 0;
    int32_t beam_size  = -1;

    bool translate     = false;
    bool no_fallback   = false;
    bool print_special = false;
    bool no_context    = true;
    bool no_timestamps = false;
    bool tinydiarize   = false;
    bool use_gpu       = true;
    bool flash_attn    = false;
    bool use_stdin     = false; // read audio from stdin instead of microphone
    bool vad           = false;

    float vad_threshold = 0.5f;

    std::string language  = "en";
    std::string model     = "models/large-v3-turbo-q5_0.bin";
    std::string vad_model = "";
};

void whisper_print_usage(int argc, char ** argv, const whisper_params & params);

// Whisper emits annotations such as "[BLANK_AUDIO]" or "(music)" for non-speech audio.
static bool is_non_speech_annotation(const char * text) {
    std::string s(text);
    const size_t first = s.find_first_not_of(" \t\n");
    if (first == std::string::npos) {
        return true;
    }
    const size_t last = s.find_last_not_of(" \t\n");
    const char open = s[first], close = s[last];
    return (open == '[' && close == ']') || (open == '(' && close == ')');
}

static bool whisper_params_parse(int argc, char ** argv, whisper_params & params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            whisper_print_usage(argc, argv, params);
            exit(0);
        }
        else if (arg == "-t"    || arg == "--threads")       { params.n_threads     = std::stoi(argv[++i]); }
        else if (                  arg == "--length")        { params.length_ms     = std::stoi(argv[++i]); }
        else if (                  arg == "--keep")          { params.keep_ms       = std::stoi(argv[++i]); }
        else if (arg == "-mt"   || arg == "--max-tokens")    { params.max_tokens    = std::stoi(argv[++i]); }
        else if (arg == "-ac"   || arg == "--audio-ctx")     { params.audio_ctx     = std::stoi(argv[++i]); }
        else if (arg == "-bs"   || arg == "--beam-size")     { params.beam_size     = std::stoi(argv[++i]); }
        else if (arg == "-tr"   || arg == "--translate")     { params.translate     = true; }
        else if (arg == "-nf"   || arg == "--no-fallback")   { params.no_fallback   = true; }
        else if (arg == "-ps"   || arg == "--print-special") { params.print_special = true; }
        else if (arg == "-kc"   || arg == "--keep-context")  { params.no_context    = false; }
        else if (arg == "-l"    || arg == "--language")      { params.language      = argv[++i]; }
        else if (arg == "-m"    || arg == "--model")         { params.model         = argv[++i]; }
        else if (arg == "-tdrz" || arg == "--tinydiarize")   { params.tinydiarize   = true; }
        else if (arg == "-ng"   || arg == "--no-gpu")        { params.use_gpu       = false; }
        else if (arg == "-fa"   || arg == "--flash-attn")    { params.flash_attn    = true; }
        else if (arg == "-stdin"|| arg == "--stdin")         { params.use_stdin     = true; }
        else if (                  arg == "--vad")           { params.vad           = true; }
        else if (arg == "-vm"   || arg == "--vad-model")     { params.vad_model     = argv[++i]; }
        else if (arg == "-vt"   || arg == "--vad-threshold") { params.vad_threshold = std::stof(argv[++i]); }

        else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            whisper_print_usage(argc, argv, params);
            exit(0);
        }
    }

    return true;
}

void whisper_print_usage(int /*argc*/, char ** argv, const whisper_params & params) {
    fprintf(stderr, "\n");
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h,       --help          [default] show this help message and exit\n");
    fprintf(stderr, "  -t N,     --threads N     [%-7d] number of threads to use during computation\n",    params.n_threads);
    fprintf(stderr, "            --length N      [%-7d] audio length in milliseconds\n",                   params.length_ms);
    fprintf(stderr, "            --keep N        [%-7d] audio to keep from previous step in ms\n",         params.keep_ms);
    fprintf(stderr, "  -mt N,    --max-tokens N  [%-7d] maximum number of tokens per audio chunk\n",       params.max_tokens);
    fprintf(stderr, "  -ac N,    --audio-ctx N   [%-7d] audio context size (0 - all)\n",                   params.audio_ctx);
    fprintf(stderr, "  -bs N,    --beam-size N   [%-7d] beam size for beam search\n",                      params.beam_size);
    fprintf(stderr, "  -tr,      --translate     [%-7s] translate from source language to english\n",      params.translate ? "true" : "false");
    fprintf(stderr, "  -nf,      --no-fallback   [%-7s] do not use temperature fallback while decoding\n", params.no_fallback ? "true" : "false");
    fprintf(stderr, "  -ps,      --print-special [%-7s] print special tokens\n",                           params.print_special ? "true" : "false");
    fprintf(stderr, "  -kc,      --keep-context  [%-7s] keep context between audio chunks\n",              params.no_context ? "false" : "true");
    fprintf(stderr, "  -l LANG,  --language LANG [%-7s] spoken language\n",                                params.language.c_str());
    fprintf(stderr, "  -m FNAME, --model FNAME   [%-7s] model path\n",                                     params.model.c_str());
    fprintf(stderr, "  -tdrz,    --tinydiarize   [%-7s] enable tinydiarize (requires a tdrz model)\n",     params.tinydiarize ? "true" : "false");
    fprintf(stderr, "  -ng,      --no-gpu        [%-7s] disable GPU inference\n",                          params.use_gpu ? "false" : "true");
    fprintf(stderr, "  -fa,      --flash-attn    [%-7s] flash attention during inference\n",               params.flash_attn ? "true" : "false");
    fprintf(stderr, "  -stdin,   --stdin         [%-7s] read audio data from stdin\n",                     params.use_stdin ? "true" : "false");
    fprintf(stderr, "            --vad           [%-7s] skip non-speech audio with voice activity detection\n", params.vad ? "true" : "false");
    fprintf(stderr, "  -vm FNAME, --vad-model FNAME [%-7s] VAD model path (required with --vad)\n",        params.vad_model.c_str());
    fprintf(stderr, "  -vt N,    --vad-threshold N [%-7.2f] speech probability threshold\n",               params.vad_threshold);
    fprintf(stderr, "\n");
}

bool validate_params(const whisper_params & params) {
    if (params.n_threads <= 0) {
        fprintf(stderr, "error: number of threads must be greater than 0\n");
        return false;
    }
    if (params.length_ms <= 0) {
        fprintf(stderr, "error: audio length must be greater than 0 ms\n");
        return false;
    }
    if (params.keep_ms < 0) {
        fprintf(stderr, "error: audio keep length cannot be negative\n");
        return false;
    }
    if (params.max_tokens < 0) {
        fprintf(stderr, "error: max tokens cannot be negative\n");
        return false;
    }
    if (params.audio_ctx < 0) {
        fprintf(stderr, "error: audio context size cannot be negative\n");
        return false;
    }
    if (params.beam_size < -1) {
        fprintf(stderr, "error: beam size must be -1 (disabled) or greater than or equal to 0\n");
        return false;
    }
    if (params.length_ms > 30000) {
        fprintf(stderr, "error: audio length cannot exceed 30000 ms (30 seconds)\n");
        return false;
    }
    if (params.keep_ms > params.length_ms) {
        fprintf(stderr, "error: audio keep length cannot exceed audio length\n");
        return false;
    }
    if (params.vad && params.vad_model.empty()) {
        fprintf(stderr, "error: --vad requires --vad-model\n");
        return false;
    }
    if (params.vad_threshold < 0.0f || params.vad_threshold > 1.0f) {
        fprintf(stderr, "error: VAD threshold must be between 0 and 1\n");
        return false;
    }
    return true;
}

int main(int argc, char ** argv) {
    whisper_params params;
    if (whisper_params_parse(argc, argv, params) == false || validate_params(params) == false) {
        return 1;
    }

    const int n_samples_len  = (1e-3*params.length_ms)*WHISPER_SAMPLE_RATE;
    const int n_samples_keep = (1e-3*params.keep_ms  )*WHISPER_SAMPLE_RATE;
    const int n_samples_30s  = (1e-3*30000.0         )*WHISPER_SAMPLE_RATE;
    params.no_timestamps  = true;
    params.max_tokens     = 0;

    // whisper init
    if (params.language != "auto" && whisper_lang_id(params.language.c_str()) == -1){
        fprintf(stderr, "error: unknown language '%s'\n", params.language.c_str());
        whisper_print_usage(argc, argv, params);
        exit(0);
    }
    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu    = params.use_gpu;
    cparams.flash_attn = params.flash_attn;
    struct whisper_context * ctx = whisper_init_from_file_with_params(params.model.c_str(), cparams);

    std::vector<whisper_token> prompt_tokens;
    whisper_full_params wparams = whisper_full_default_params(params.beam_size > 1 ? WHISPER_SAMPLING_BEAM_SEARCH : WHISPER_SAMPLING_GREEDY);

    wparams.print_progress   = false;
    wparams.print_special    = params.print_special;
    wparams.print_realtime   = false;
    wparams.print_timestamps = !params.no_timestamps;
    wparams.translate        = params.translate;
    wparams.max_tokens       = params.max_tokens;
    wparams.language         = params.language.c_str();
    wparams.n_threads        = params.n_threads;
    wparams.beam_search.beam_size = params.beam_size;
    wparams.audio_ctx        = params.audio_ctx;
    wparams.tdrz_enable      = params.tinydiarize;
    wparams.vad              = params.vad;
    wparams.vad_model_path   = params.vad ? params.vad_model.c_str() : nullptr;
    wparams.vad_params       = whisper_vad_default_params();
    wparams.vad_params.threshold = params.vad_threshold;
    wparams.temperature_inc  = params.no_fallback ? 0.0f : wparams.temperature_inc;
    wparams.prompt_tokens    = prompt_tokens.data();
    wparams.prompt_n_tokens  = prompt_tokens.size();

    // print some info about the processing
    fprintf(stderr, "\n");
    if (!whisper_is_multilingual(ctx)) {
        if (params.language != "en" || params.translate) {
            params.language = "en";
            params.translate = false;
            fprintf(stderr, "%s: WARNING: model is not multilingual, ignoring language and translation options\n", __func__);
        }
    }
    fprintf(stderr, "%s: processing %d samples ( len = %.1f sec / keep = %.1f sec / sample_rate = %d), %d threads, lang = %s, task = %s, timestamps = %d ...\n",
            __func__,
            n_samples_len,
            float(n_samples_len )/WHISPER_SAMPLE_RATE,
            float(n_samples_keep)/WHISPER_SAMPLE_RATE,
            WHISPER_SAMPLE_RATE,
            params.n_threads,
            params.language.c_str(),
            params.translate ? "translate" : "transcribe",
            params.no_timestamps ? 0 : 1);
    fprintf(stderr, "\n");

    // audio buffer and other variables
    float pcmf32[n_samples_30s];
    int pcmf32_len = 0, i, j, n_segments, token_count, segments_to_copy, bytes_to_copy;
    size_t b_read;
    float pcmf32_temp[n_samples_keep];
    bool markerFound = false, should_process = false;
    const char * text;

    // main audio loop
    while (true) {

        b_read = read(0, 
                      reinterpret_cast<char*>(pcmf32 + pcmf32_len),
                      (n_samples_30s - pcmf32_len) * sizeof(float));
        
        if (b_read <= 0) {perror("read");exit(1);}

        pcmf32_len += b_read / sizeof(float);

        // Check for marker NaN, decide if we should process, etc.
        if (pcmf32_len > 0 && std::isnan(pcmf32[pcmf32_len - 1])) {
            fprintf(stderr, "Marker found\n");
            markerFound = true;
            pcmf32_len--;
        }
        
        if (markerFound || pcmf32_len >= n_samples_len) {
            should_process = true;
        }

        // An utterance closed with almost no audio (client left right away) has nothing to transcribe
        if (markerFound && pcmf32_len < WHISPER_SAMPLE_RATE / 10) {
            fprintf(stderr, "Skipping %d audio samples (too short)\n", pcmf32_len);
            markerFound = false;
            should_process = false;
            prompt_tokens.clear();
            pcmf32_len = 0;
            printf("\n<|endoftext|>\n");
            fflush(stdout);
            continue;
        }

        if (should_process) {
            fprintf(stderr, "Processing %d audio samples (%.1f s)\n", pcmf32_len, (float) pcmf32_len / WHISPER_SAMPLE_RATE);

            // Whisper inference
            if (whisper_full(ctx, wparams, pcmf32, pcmf32_len) != 0) {
                fprintf(stderr, "%s: failed to process audio\n", argv[0]);
                return 1;
            }

            n_segments = whisper_full_n_segments(ctx);
            if (params.vad) {
                fprintf(stderr, "VAD speech segments: %d\n", whisper_full_n_vad_segments(ctx));
            }

            // Print in stdoud
            for (i = 0; i < n_segments; i++) {
                text = whisper_full_get_segment_text(ctx, i);

                if (is_non_speech_annotation(text)) {
                    fprintf(stderr, "whisper transcript segment %d skipped (non-speech): %s\n", i, text);
                    continue;
                }
                printf("%s", text);
                fprintf(stderr, "whisper transcript segment %d: %s\n",i, text);
                fflush(stdout);
            }

            if (markerFound) {
                markerFound = false;
                should_process = false;
                prompt_tokens.clear();
                pcmf32_len = 0;
                // Print end of text marker
                printf("\n<|endoftext|>\n");
                fflush(stdout);
            
            } else {

                // Add tokens of the last full length segment as the prompt
                if (!params.no_context) {
                    prompt_tokens.clear();
                    for (i = 0; i < n_segments; i++) {
                        token_count = whisper_full_n_tokens(ctx, i);
                        for (j = 0; j < token_count; j++) {
                            prompt_tokens.push_back(whisper_full_get_token_id(ctx, i, j));
                        }
                    }
                    wparams.prompt_n_tokens  = prompt_tokens.size();
                }

                // keep part of the audio for next iteration to try to mitigate word boundary issues
                segments_to_copy = std::min(n_samples_keep, pcmf32_len);
                bytes_to_copy = segments_to_copy * sizeof(float);
                memcpy(pcmf32_temp, pcmf32 + pcmf32_len - segments_to_copy, bytes_to_copy);
                memcpy(pcmf32, pcmf32_temp, bytes_to_copy);
                pcmf32_len = segments_to_copy;

                should_process = false;
            }
        }
    }

    whisper_print_timings(ctx);
    whisper_free(ctx);

    return 0;
}