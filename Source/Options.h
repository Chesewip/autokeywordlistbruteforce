#pragma once
#include <string>
#include <thread>
#include <iostream>

struct Options 
{
    std::string ciphertext;
    std::string wordlist;
    std::string alphabet_wordlist;
    std::string two_letter_list;
    std::string spacing_wordlist;
    std::string spacing_guide;
    std::size_t max_results = 50;
    std::size_t preview_length = 80;
    std::size_t threads = std::max<std::size_t>(1, std::thread::hardware_concurrency());
    bool include_autokey = false;
    double progress_interval_seconds = 1.0;
    bool quiet = false;
    bool use_cuda = true;
    bool include_keyword_front = true;
    bool include_reversed_keyword_front = true;
    bool include_keyword_back = true;
    bool include_keyword_front_reversed_alphabet = true;
    bool include_keyword_back_reversed_alphabet = true;


    static std::string read_file(const std::string& path) 
    {
        std::ifstream input(path);
        if (!input) {
            throw std::runtime_error("Failed to open file: " + path);
        }
        std::ostringstream buffer;
        buffer << input.rdbuf();
        return buffer.str();
    }


    static Options parse_options(int argc, char* argv[])
    {
        Options options;
        for (int i = 1; i < argc; ++i) 
        {
            std::string arg = argv[i];
            auto require_value = [&](const std::string& name) -> std::string {
                if (i + 1 >= argc) {
                    throw std::runtime_error("Missing value for option " + name);
                }
                return argv[++i];
            };

            if (arg == "--ciphertext") {
                options.ciphertext = require_value(arg);
            }
            else if (arg == "--ciphertext-file") {
                options.ciphertext = read_file(require_value(arg));
            }
            else if (arg == "--wordlist") {
                options.wordlist = read_file(require_value(arg));
            }
            else if (arg == "--wordlist-inline") {
                options.wordlist = require_value(arg);
            }
            else if (arg == "--alphabet-wordlist") {
                options.alphabet_wordlist = read_file(require_value(arg));
            }
            else if (arg == "--alphabet-wordlist-inline") {
                options.alphabet_wordlist = require_value(arg);
            }
            else if (arg == "--two-letter-list") {
                options.two_letter_list = read_file(require_value(arg));
            }
            else if (arg == "--two-letter-inline") {
                options.two_letter_list = require_value(arg);
            }
            else if (arg == "--spacing-wordlist") {
                options.spacing_wordlist = read_file(require_value(arg));
            }
            else if (arg == "--spacing-wordlist-inline") {
                options.spacing_wordlist = require_value(arg);
            }
            else if (arg == "--spacing-guide") {
                options.spacing_guide = require_value(arg);
            }
            else if (arg == "--spacing-guide-file") {
                options.spacing_guide = read_file(require_value(arg));
            }
            else if (arg == "--max-results") {
                options.max_results = static_cast<std::size_t>(std::stoul(require_value(arg)));
            }
            else if (arg == "--preview-length") {
                options.preview_length = static_cast<std::size_t>(std::stoul(require_value(arg)));
            }
            else if (arg == "--threads") {
                options.threads = static_cast<std::size_t>(std::stoul(require_value(arg)));
            }
            else if (arg == "--include-autokey") {
                options.include_autokey = true;
            }
            else if (arg == "--progress-interval") {
                options.progress_interval_seconds = std::stod(require_value(arg));
            }
            else if (arg == "--quiet") {
                options.quiet = true;
            }
            else if (arg == "--use-cuda") {
                options.use_cuda = true;
            }
            else if (arg == "--no-cuda") {
                options.use_cuda = false;
            }
            else if (arg == "--no-keyword-front") {
                options.include_keyword_front = false;
            }
            else if (arg == "--no-reversed-keyword-front") {
                options.include_reversed_keyword_front = false;
            }
            else if (arg == "--no-keyword-back") {
                options.include_keyword_back = false;
            }
            else if (arg == "--no-keyword-front-reversed") {
                options.include_keyword_front_reversed_alphabet = false;
            }
            else if (arg == "--no-keyword-back-reversed") {
                options.include_keyword_back_reversed_alphabet = false;
            }
            else if (arg == "--help" || arg == "-h") {
                std::cout << "Quagmire III wordlist bruteforcer (C++)\n"
                    << "Options:\n"
                    << "  --ciphertext <text>             Inline ciphertext string\n"
                    << "  --ciphertext-file <path>        File containing ciphertext\n"
                    << "  --wordlist <path>               Main wordlist file (required)\n"
                    << "  --wordlist-inline <text>        Inline wordlist string\n"
                    << "  --alphabet-wordlist <path>      Alphabet wordlist file (defaults to main)\n"
                    << "  --alphabet-wordlist-inline <text> Inline alphabet wordlist string\n"
                    << "  --two-letter-list <path>        Optional 2-letter filter list\n"
                    << "  --two-letter-inline <text>      Inline 2-letter filter string\n"
                    << "  --spacing-wordlist <path>       Wordlist used for spacing guide scoring\n"
                    << "  --spacing-wordlist-inline <text> Inline spacing wordlist string\n"
                    << "  --spacing-guide <pattern>       Word length pattern (e.g. 2-4-3-3)\n"
                    << "  --spacing-guide-file <path>     File containing word length pattern\n"
                    << "  --max-results <N>               Max candidates to keep (default 50)\n"
                    << "  --preview-length <N>            Plaintext preview length (default 80)\n"
                    << "  --threads <N>                   Worker threads (default hardware)\n"
                    << "  --include-autokey               Try autokey variants too\n"
                    << "  --use-cuda                      Enable experimental CUDA execution\n"
                    << "  --no-cuda                       Force CPU execution (default)\n"
                    << "  --progress-interval <sec>       Progress update interval (default 1.0)\n"
                    << "  --quiet                         Suppress periodic progress output\n"
                    << "  --no-keyword-front              Skip key prefix on normal alphabet\n"
                    << "  --no-reversed-keyword-front     Skip reversed key prefix variant\n"
                    << "  --no-keyword-back               Skip key suffix on normal alphabet\n"
                    << "  --no-keyword-front-reversed     Skip key prefix on reversed alphabet\n"
                    << "  --no-keyword-back-reversed      Skip key suffix on reversed alphabet\n"
                    << "  --help                          Show this help message\n";
                std::exit(0);
            }
            else {
                throw std::runtime_error("Unknown option: " + arg);
            }
        }

        if (options.ciphertext.empty()) {
            throw std::runtime_error("Ciphertext is required (use --ciphertext or --ciphertext-file)");
        }
        if (options.wordlist.empty()) {
            throw std::runtime_error("Wordlist is required (use --wordlist or --wordlist-inline)");
        }
        if (options.alphabet_wordlist.empty()) {
            options.alphabet_wordlist = options.wordlist;
        }
        if (!options.spacing_guide.empty() && options.spacing_wordlist.empty()) {
            throw std::runtime_error(
                "Spacing guide provided but spacing wordlist is missing (use --spacing-wordlist)");
        }
        if (options.threads == 0) {
            options.threads = 1;
        }
        if (!options.include_keyword_front && !options.include_reversed_keyword_front &&
            !options.include_keyword_back &&
            !options.include_keyword_front_reversed_alphabet &&
            !options.include_keyword_back_reversed_alphabet) {
            throw std::runtime_error(
                "All alphabet construction variants are disabled. Enable at least one option.");
        }
        return options;
    }

};


