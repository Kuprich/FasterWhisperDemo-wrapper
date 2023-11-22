# Input Arguments
from argparse import ArgumentParser


def GetArguments():

    parser = CustomArgumentParser()

    parser.AddWhisperModelArguments()
    parser.AddTranscrbeArguments()

    return parser.parse_args()


class CustomArgumentParser(ArgumentParser):

    def AddWhisperModelArguments(self):
        model_parser = self.add_argument_group("Faster Whisper Model Argumets")

        model_parser.add_argument(
            "--model_size_or_path",
            default="base",
            help="""Size of the model to use (tiny, tiny.en, base, base.en,
                    small, small.en, medium, medium.en, large-v1, large-v2, or large), a path to a converted
                    model directory, or a CTranslate2-converted Whisper model ID from the Hugging Face Hub.
                    When a size or a model ID is configured, the converted model is downloaded
                    from the Hugging Face Hub.""",
        )

        model_parser.add_argument(
            "--device",
            default="auto",
            help='Device to use for computation ("cpu", "cuda", "auto").',
        )

        model_parser.add_argument(
            "--device_index",
            type=int,
            nargs="+",
            default=0,
            help="""Device ID to use. 
                    The model can also be loaded on multiple GPUs by passing a list of IDs
                    (e.g. [0, 1, 2, 3]). In that case, multiple transcriptions can run in parallel
                    when transcribe() is called from multiple Python threads (see also num_workers).""",
        )

        model_parser.add_argument(
            "--compute_type",
            default="default",
            help="""Type to use for computation.
                    See https://opennmt.net/CTranslate2/quantization.html.""",
        )

        model_parser.add_argument(
            "--cpu_threads",
            type=int,
            default=0,
            help="""TNumber of threads to use when running on CPU (4 by default).
                    A non zero value overrides the OMP_NUM_THREADS environment variable.""",
        )

        model_parser.add_argument(
            "--num_workers",
            type=int,
            default=1,
            help="""When transcribe() is called from multiple Python threads,
                    having multiple workers enables true parallelism when running the model
                    (concurrent calls to self.model.generate() will run in parallel).
                    This can improve the global throughput at the cost of increased memory usage.""",
        )

        model_parser.add_argument(
            "--download_root",
            help="""Directory where the models should be saved. If not set, the models
                    are saved in the standard Hugging Face cache directory.""",
        )

        model_parser.add_argument(
            "--local_files_only",
            help="""If True, avoid downloading the file and return the path to the
                    local cached file if it exists""",
            action="store_true",
        )

    def AddTranscrbeArguments(self):

        transcribe_parser = self.add_argument_group("Transcribe Method Arguments")

        transcribe_parser.add_argument(
            "--audio",
            help="Path to the input file (or a file-like object), or the audio waveform.",
        )

        transcribe_parser.add_argument(
            "--language",
            help="""The language spoken in the audio. It should be a language code such
            as "en" or "fr". If not set, the language will be detected in the first 30 seconds
            of audio.""",
        )

        transcribe_parser.add_argument(
            "--task",
            help="Task to execute (transcribe or translate).",
        )

        transcribe_parser.add_argument(
            "--beam_size",
            help="Beam size to use for decoding",
        )

        transcribe_parser.add_argument(
            "--best_of",
            help="Number of candidates when sampling with non-zero temperature.",
        )

        transcribe_parser.add_argument(
            "--patience",
            help="Beam search patience factor",
        )

        transcribe_parser.add_argument(
            "--length_penalty",
            help="Exponential length penalty constant.",
        )

        transcribe_parser.add_argument(
            "--repetition_penalty",
            help="""Penalty applied to the score of previously generated tokens
            (set > 1 to penalize).""",
        )

        transcribe_parser.add_argument(
            "--no_repeat_ngram_size",
            help="Prevent repetitions of ngrams with this size (set 0 to disable)",
        )

        transcribe_parser.add_argument(
            "--temperature",
            help="""Temperature for sampling. It can be a tuple of temperatures,
            which will be successively used upon failures according to either
            `compression_ratio_threshold` or `log_prob_threshold`.""",
        )

        transcribe_parser.add_argument(
            "--compression_ratio_threshold",
            help="""If the gzip compression ratio is above this value,
            treat as failed.""",
        )

        transcribe_parser.add_argument(
            "--log_prob_threshold",
            help="""If the average log probability over sampled tokens is
            below this value, treat as failed.""",
        )

        transcribe_parser.add_argument(
            "--no_speech_threshold",
            help="""If the no_speech probability is higher than this value AND
            the average log probability over sampled tokens is below `log_prob_threshold`,
            consider the segment as silent.""",
        )

        transcribe_parser.add_argument(
            "--condition_on_previous_text",
            help="""If True, the previous output of the model is provided
            as a prompt for the next window; disabling may make the text inconsistent across
            windows, but the model becomes less prone to getting stuck in a failure loop,
            such as repetition looping or timestamps going out of sync.""",
        )

        transcribe_parser.add_argument(
            "--prompt_reset_on_temperature",
            help="""Resets prompt if temperature is above this value.
            Arg has effect only if condition_on_previous_text is True.""",
        )

        transcribe_parser.add_argument(
            "--initial_prompt",
            help="""Optional text string or iterable of token ids to provide as a
            prompt for the first window""",
        )

        transcribe_parser.add_argument(
            "--prefix",
            help="""Optional text to provide as a prefix for the first window.""",
        )

        transcribe_parser.add_argument(
            "--suppress_blank",
            help="""Suppress blank outputs at the beginning of the sampling.""",
        )

        transcribe_parser.add_argument(
            "--suppress_tokens",
            help="""List of token IDs to suppress. -1 will suppress a default set
            of symbols as defined in the model config.json file.""",
        )

        transcribe_parser.add_argument(
            "--without_timestamps",
            help="""Only sample text tokens""",
        )

        transcribe_parser.add_argument(
            "--max_initial_timestamp",
            help="""The initial timestamp cannot be later than this.""",
        )

        transcribe_parser.add_argument(
            "--word_timestamps",
            help="""Extract word-level timestamps using the cross-attention pattern
            and dynamic time warping, and include the timestamps for each word in each segment.""",
        )

        transcribe_parser.add_argument(
            "--prepend_punctuations",
            help="""If word_timestamps is True, merge these punctuation symbols
            with the next word""",
        )

        transcribe_parser.add_argument(
            "--append_punctuations",
            help="""If word_timestamps is True, merge these punctuation symbols
            with the previous word""",
        )

        transcribe_parser.add_argument(
            "--vad_filter",
            help="""nable the voice activity detection (VAD) to filter out parts of the audio
            without speech. This step is using the Silero VAD model
            https://github.com/snakers4/silero-vad.""",
        )

        transcribe_parser.add_argument(
            "--vad_parameters",
            help="""Dictionary of Silero VAD parameters or VadOptions class (see available
            parameters and default values in the class `VadOptions`).""",
        )



GetArguments()