#pragma once
#include <string>
#include <array>
#include <vector>
#include <map>
#include "Options.h"

struct AlphabetCandidate 
{
  std::string alphabet;
  std::array<int, 26> index_map{};
  std::string base_word;
  bool keyword_reversed = false;
  bool alphabet_reversed = false;
  bool keyword_front = true;
};


struct AlphabetBuilder
{
    static std::string build_keyed_alphabet(const std::string& word,
        bool keyword_reversed,
        bool alphabet_reversed,
        bool keyword_front,
        int indicator_shift = 1) // -1 = no shift
    {
        std::array<bool, 26> seen{};
        std::string ordered_word;

        if (keyword_reversed)
            ordered_word.assign(word.rbegin(), word.rend());
        else
            ordered_word = word;

        // Build unique-key from the keyword
        std::string unique_key;
        unique_key.reserve(ordered_word.size());
        for (char ch : ordered_word)
        {
            int idx = ch - 'A';
            if (idx >= 0 && idx < 26 && !seen[idx])
            {
                seen[idx] = true;
                unique_key.push_back(ch);
            }
        }

        // Build base alphabet
        std::string alphabet;
        alphabet.reserve(26);

        if (keyword_front)
            alphabet.append(unique_key);

        if (alphabet_reversed)
        {
            for (int i = 25; i >= 0; --i)
            {
                if (!seen[i])
                    alphabet.push_back(static_cast<char>('A' + i));
            }
        }
        else
        {
            for (int i = 0; i < 26; ++i)
            {
                if (!seen[i])
                    alphabet.push_back(static_cast<char>('A' + i));
            }
        }

        if (!keyword_front)
            alphabet.append(unique_key);

        // Optional indicator / shift: rotate the final alphabet
        // indicator_shift > 0 => rotate left by that many positions
        if (indicator_shift != -1 && !alphabet.empty())
        {
            const int n = static_cast<int>(alphabet.size());
            int s = indicator_shift % n;
            if (s < 0)
                s += n; // support negative shifts if you ever want them

            if (s != 0)
                std::rotate(alphabet.begin(), alphabet.begin() + s, alphabet.end());
        }

        return alphabet;
    }

    static AlphabetCandidate make_alphabet_candidate(const std::string& word,
        bool keyword_reversed,
        bool alphabet_reversed,
        bool keyword_front,
        const int shift = -1) 
    {
        AlphabetCandidate candidate;
        candidate.base_word = word;
        candidate.keyword_reversed = keyword_reversed;
        candidate.alphabet_reversed = alphabet_reversed;
        candidate.keyword_front = keyword_front;
        candidate.alphabet = build_keyed_alphabet(word, keyword_reversed, alphabet_reversed, keyword_front,shift);
        candidate.index_map.fill(-1);
        for (std::size_t i = 0; i < candidate.alphabet.size(); ++i) 
        {
            char ch = candidate.alphabet[i];
            candidate.index_map[ch - 'A'] = static_cast<int>(i);
        }
        return candidate;
    }

    static std::vector<AlphabetCandidate> build_alphabet_candidates(const std::vector<std::string>& words, const Options& options)
    {
        std::map<std::string, AlphabetCandidate> unique_map;
        for (auto i = 0; i < 1; i++)
        {
            for (const auto& word : words) {
                if (options.include_keyword_front) {
                    AlphabetCandidate forward =
                        make_alphabet_candidate(word, false, false, true, i);
                    unique_map.emplace(forward.alphabet, forward);
                }
                if (options.include_reversed_keyword_front) {
                    AlphabetCandidate reversed_key =
                        make_alphabet_candidate(word, true, false, true, i);
                    unique_map.emplace(reversed_key.alphabet, reversed_key);
                }
                if (options.include_keyword_back) {
                    AlphabetCandidate back = make_alphabet_candidate(word, false, false, false, i);
                    unique_map.emplace(back.alphabet, back);
                }
                if (options.include_keyword_front_reversed_alphabet) {
                    AlphabetCandidate rev_alphabet_front =
                        make_alphabet_candidate(word, false, true, true, i);
                    unique_map.emplace(rev_alphabet_front.alphabet, rev_alphabet_front);
                }
                if (options.include_keyword_back_reversed_alphabet) {
                    AlphabetCandidate rev_alphabet_back =
                        make_alphabet_candidate(word, false, true, false, i);
                    unique_map.emplace(rev_alphabet_back.alphabet, rev_alphabet_back);
                }
                if (true) { //Reversed alphabet and reverse key forgot to add, just always true for now
                    AlphabetCandidate rev_alphabet_back =
                        make_alphabet_candidate(word, true, true, false, i);
                    unique_map.emplace(rev_alphabet_back.alphabet, rev_alphabet_back);
                }
                if (true) { //Reversed alphabet and reverse key forgot to add, just always true for now
                    AlphabetCandidate rev_alphabet_back =
                        make_alphabet_candidate(word, true, true, true, i);
                    unique_map.emplace(rev_alphabet_back.alphabet, rev_alphabet_back);
                }
                if (true) { //Reversed alphabet and reverse key forgot to add, just always true for now
                    AlphabetCandidate rev_alphabet_back =
                        make_alphabet_candidate(word, true, false, false, i);
                    unique_map.emplace(rev_alphabet_back.alphabet, rev_alphabet_back);
                }
            }
        }
        
        std::vector<AlphabetCandidate> result;
        result.reserve(unique_map.size());
        for (auto& entry : unique_map) {
            result.push_back(std::move(entry.second));
        }
        return result;
    }

    static inline int alphabet_index(const AlphabetCandidate& alphabet, char ch)
    {
        if (ch < 'A' || ch > 'Z') {
            return -1;
        }
        return alphabet.index_map[ch - 'A'];
    }

};