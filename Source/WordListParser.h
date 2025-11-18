#pragma once 
#include <string>
#include <vector>
#include <iostream>
#include <unordered_set>
#include <sstream>
#include <unordered_map>

struct WordlistParser
{

    static std::string clean_letters(const std::string& text) 
    {
        std::string result;
        result.reserve(text.size());
        for (char ch : text) {
            if (ch >= 'A' && ch <= 'Z') {
                result.push_back(ch);
            }
            else if (ch >= 'a' && ch <= 'z') {
                result.push_back(static_cast<char>('A' + (ch - 'a')));
            }
        }
        return result;
    }

    // Build an "interrupted" key by restarting the key at each word boundary
    // defined by spacing_pattern. Each segment of length L takes the first L
    // characters of the key, repeating the key as needed.
    static std::string build_interrupted_key(const std::string& key,
        const std::vector<int>& spacing_pattern)
    {
        if (key.size() < 6u)  // your constraint: need at least a 6-letter key
            return {};

        std::size_t total_len = 0;
        for (int len : spacing_pattern)
            if (len > 0)
                total_len += static_cast<std::size_t>(len);

        if (total_len == 0)
            return {};

        std::string out;
        out.reserve(total_len);

        for (int len : spacing_pattern)
        {
            if (len <= 0)
                continue;

            // For each word of length len, restart the key
            for (int i = 0; i < len; ++i)
            {
                // repeat key within the word if the word is longer than key
                out.push_back(key[static_cast<std::size_t>(i) % key.size()]);
            }
        }

        return out;
    }


    static std::vector<std::string> parse_wordlist(const std::string& text, const std::vector<int>* spacing_pattern = nullptr)
    {
        std::vector<std::string> words;
        std::string current;
        std::istringstream stream(text);

        const bool have_spacing = (spacing_pattern != nullptr &&
            !spacing_pattern->empty());

        while (std::getline(stream, current))
        {
            auto cleaned = clean_letters(current);
            if (cleaned.empty())
                continue;

            // Always keep the base key
            words.push_back(std::move(cleaned));
            const std::string& base = words.back();

            // For keys >= 6 letters, add the interrupted variant if spacing is available
            /*if (have_spacing && base.size() >= 6u)
            {
                std::string interrupted = build_interrupted_key(base, *spacing_pattern);
                if (!interrupted.empty())
                {
                    words.push_back(std::move(interrupted));
                }
            }*/
        }

        return words;
    }
    static inline std::uint16_t encode_bigram(char a, char b)
    {
        return static_cast<std::uint16_t>((a - 'A') * 26 + (b - 'A'));
    }

    static inline std::uint32_t encode_quad(char a, char b, char c, char d)
    {
        // assume A-Z; caller should validate if needed
        int ia = a - 'A';
        int ib = b - 'A';
        int ic = c - 'A';
        int id = d - 'A';

        // (((a * 26) + b) * 26 + c) * 26 + d
        return static_cast<std::uint32_t>(
            (((ia * 26 + ib) * 26 + ic) * 26 + id)
            );
    }

    static std::unordered_set<std::uint16_t> parse_two_letter_list(const std::string& text)
    {
        std::unordered_set<std::uint16_t> pairs;

        std::string current;
        std::istringstream stream(text);

        while (std::getline(stream, current)) {
            auto cleaned = clean_letters(current);  // already uppercases
            if (cleaned.size() == 2) {
                char a = cleaned[0];
                char b = cleaned[1];

                // Optional: be defensive, skip non A-Z
                if (a < 'A' || a > 'Z' || b < 'A' || b > 'Z') {
                    continue;
                }

                pairs.insert(encode_bigram(a, b));
            }
        }

        return pairs;
    }


    static std::unordered_set<std::uint32_t> build_four_letter_set(const std::vector<std::string>& words)
    {
        std::unordered_set<std::uint32_t> result;
        for (const auto& w : words) {
            if (w.size() == 4) {
                char a = w[0];
                char b = w[1];
                char c = w[2];
                char d = w[3];

                // be defensive if you like
                if (a < 'A' || a > 'Z' ||
                    b < 'A' || b > 'Z' ||
                    c < 'A' || c > 'Z' ||
                    d < 'A' || d > 'Z') {
                    continue;
                }

                result.insert(encode_quad(a, b, c, d));
            }
        }
        return result;
    }

    static std::unordered_map<int, std::unordered_set<std::string>> build_words_by_length(const std::vector<std::string>& words)
    {
        std::unordered_map<int, std::unordered_set<std::string>> grouped;
        for (const auto& word : words) {
            grouped[static_cast<int>(word.size())].insert(word);
        }
        return grouped;
    }

    static std::vector<int> parse_spacing_pattern(const std::string& pattern_text)
    {
        std::vector<int> pattern;
        int current = 0;
        bool in_number = false;
        for (char ch : pattern_text) {
            if (std::isdigit(static_cast<unsigned char>(ch))) {
                current = current * 10 + (ch - '0');
                in_number = true;
            }
            else {
                if (in_number) {
                    if (current <= 0) {
                        throw std::runtime_error("Spacing guide values must be positive integers");
                    }
                    pattern.push_back(current);
                    current = 0;
                    in_number = false;
                }
            }
        }
        if (in_number) {
            if (current <= 0) {
                throw std::runtime_error("Spacing guide values must be positive integers");
            }
            pattern.push_back(current);
        }
        return pattern;
    }

    static std::pair<int, int> count_spacing_matches( const std::string& plaintext, const std::vector<int>& pattern, const std::unordered_map<int, std::unordered_set<std::string>>& words_by_length) 
    {
        if (pattern.empty() || words_by_length.empty()) {
            return { -1, 0 };
        }
        int matches = 0;
        int considered = 0;
        std::size_t offset = 0;
        for (int length : pattern) {
            if (length <= 0) {
                continue;
            }
            if (offset + static_cast<std::size_t>(length) > plaintext.size()) {
                break;
            }
            auto it = words_by_length.find(length);
            if (it == words_by_length.end() || it->second.empty()) {
                offset += static_cast<std::size_t>(length);
                continue;
            }
            const std::string word = plaintext.substr(offset, static_cast<std::size_t>(length));
            ++considered;
            if (it->second.find(word) != it->second.end()) {
                ++matches;
            }
            offset += static_cast<std::size_t>(length);
        }
        return { matches, considered };
    }


};