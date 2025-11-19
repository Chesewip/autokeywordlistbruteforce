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

    struct SpacingPrefixKey
    {
        std::uint16_t wordLen;    // length of the word (e.g. 4)
        std::uint8_t  prefixLen;  // 1 or 2
        std::uint8_t  c0;         // first letter 0..25
        std::uint8_t  c1;         // second letter 0..25, or 0xFF if prefixLen == 1
    };

    struct SpacingPrefixKeyHash
    {
        std::size_t operator()(const SpacingPrefixKey& k) const noexcept
        {
            // simple mix of fields
            std::size_t h = std::size_t(k.wordLen);
            h = h * 131u + std::size_t(k.prefixLen);
            h = h * 131u + std::size_t(k.c0);
            h = h * 131u + std::size_t(k.c1);
            // final avalanche
            h ^= 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
            return h;
        }
    };

    struct SpacingPrefixKeyEq
    {
        bool operator()(const SpacingPrefixKey& a,
            const SpacingPrefixKey& b) const noexcept
        {
            return a.wordLen == b.wordLen
                && a.prefixLen == b.prefixLen
                && a.c0 == b.c0
                && a.c1 == b.c1;
        }
    };

    using SpacingPrefixIndex =
        std::unordered_map<SpacingPrefixKey,
        std::vector<const std::string*>,
        SpacingPrefixKeyHash,
        SpacingPrefixKeyEq>;

    static SpacingPrefixIndex build_spacing_prefix_index(
        const std::vector<std::string>& spacing_words,
        int maxPrefixLen)
    {
        SpacingPrefixIndex index;

        if (maxPrefixLen <= 0 || spacing_words.empty())
            return index;

        if (maxPrefixLen > 2)
            maxPrefixLen = 2; // we only care about 1–2 letters for now

        // Heuristic reserve: every word contributes up to maxPrefixLen entries.
        index.reserve(spacing_words.size() * maxPrefixLen);

        for (std::uint32_t wi = 0; wi < static_cast<std::uint32_t>(spacing_words.size()); ++wi)
        {
            const std::string& w = spacing_words[wi];
            const int len = static_cast<int>(w.size());
            if (len <= 0)
                continue;

            // First letter must be A–Z (we assume parse_wordlist already cleaned,
            // but be defensive).
            char c0ch = w[0];
            if (c0ch < 'A' || c0ch > 'Z')
                continue;

            std::uint8_t c0 = static_cast<std::uint8_t>(c0ch - 'A');

            // Optional second letter
            char c1ch = 0;
            std::uint8_t c1 = 0xFF;
            if (len >= 2)
            {
                c1ch = w[1];
                if (c1ch >= 'A' && c1ch <= 'Z')
                    c1 = static_cast<std::uint8_t>(c1ch - 'A');
                else
                    c1 = 0xFF; // treat as "no second letter" for indexing
            }

            const int maxForWord = std::min(maxPrefixLen, len);

            // prefixLen = 1: index by (len, first letter)
            // prefixLen = 2: index by (len, first two letters), if we have them
            for (int prefixLen = 1; prefixLen <= maxForWord; ++prefixLen)
            {
                SpacingPrefixKey key{};
                key.wordLen = static_cast<std::uint16_t>(len);
                key.prefixLen = static_cast<std::uint8_t>(prefixLen);
                key.c0 = c0;
                key.c1 = (prefixLen >= 2 ? c1 : 0xFF);

                auto& bucket = index[key];
                bucket.push_back(&spacing_words[wi]);
            }
        }

        return index;
    }

    const int Q3_GLOBAL_MAX_PREFIX = 4;

    using PrefixCode = std::uint32_t;

    static inline PrefixCode encode_prefix(const std::uint8_t* s, int len)
    {
        PrefixCode code = 0;
        for (int i = 0; i < len; ++i)
            code |= (PrefixCode(s[i] & 0x1F) << (5 * i));
        return code;
    }

    struct GlobalPrefixIndex
    {

        static constexpr int MAX_PREFIX_LEN = 6;
        // byLen[L][code] -> list of word IDs whose canonical prefix of length L
        // encodes to `code`. L in [1..Q3_GLOBAL_MAX_PREFIX].
        using BucketMap = std::unordered_map<PrefixCode, std::vector<std::uint32_t>>;

        std::array<BucketMap, MAX_PREFIX_LEN + 1> byLen;
    };

    static GlobalPrefixIndex build_global_prefix_index( const std::vector<std::string>& words, int maxPrefixLen = GlobalPrefixIndex::MAX_PREFIX_LEN)
    {
        GlobalPrefixIndex idx;

        if (words.empty())
            return idx;

        if (maxPrefixLen <= 0)
            return idx;
        if (maxPrefixLen > GlobalPrefixIndex::MAX_PREFIX_LEN)
            maxPrefixLen = GlobalPrefixIndex::MAX_PREFIX_LEN;

        // heuristically reserve: each word contributes up to maxPrefixLen prefixes
        for (int L = 1; L <= maxPrefixLen; ++L)
            idx.byLen[L].reserve(words.size() * 2);

        for (std::uint32_t wid = 0; wid < static_cast<std::uint32_t>(words.size()); ++wid)
        {
            const std::string& w = words[wid];
            if (w.empty())
                continue;

            const int wlen = static_cast<int>(w.size());
            if (wlen <= 0)
                continue;

            PrefixCode code = 0;

            // Build prefixes incrementally: P, PL, PLA, PLAN...
            const int maxForWord = std::min(wlen, maxPrefixLen);
            for (int L = 1; L <= maxForWord; ++L)
            {
                char ch = w[static_cast<std::size_t>(L - 1)];
                if (ch < 'A' || ch > 'Z')
                {
                    // stop indexing this word on first non A–Z;
                    // you can choose to `break` or `continue` based on how strict you are
                    break;
                }

                std::uint8_t canon = static_cast<std::uint8_t>(ch - 'A'); // 0..25
                code |= (PrefixCode(canon & 0x1F) << (5 * (L - 1)));

                auto& bucketMap = idx.byLen[L];
                auto& vec = bucketMap[code];
                vec.push_back(wid);
            }
        }

        return idx;
    }

    using QuadgramMap = std::unordered_map<std::uint32_t, double>;
    using QuadgramTable = std::vector<double>; // size 26*26*26*26
    static constexpr double Q3_QUADGRAM_FLOOR_LOGP = -24.00;

    static inline std::uint32_t pack_quadgram(const std::string& gram)
    {
        // assume gram is already validated to be length 4, A–Z
        int a = gram[0] - 'A';
        int b = gram[1] - 'A';
        int c = gram[2] - 'A';
        int d = gram[3] - 'A';

        return (((std::uint32_t)a * 26u + (std::uint32_t)b) * 26u + (std::uint32_t)c) * 26u
            + (std::uint32_t)d;
    }

    static QuadgramMap load_quadgram_file(const std::string& filename, double floorLogPOut)
    {
        std::ifstream in(filename);
        if (!in) {
            throw std::runtime_error("Failed to open file: " + filename);
        }
        QuadgramMap table;

        // Default floor if we get nothing / something weird
        floorLogPOut = -24.0;

        if (!in)
            return table;

        std::string line;
        double min_logp = std::numeric_limits<double>::infinity();

        while (std::getline(in, line))
        {
            // Trim leading/trailing whitespace
            auto is_space = [](unsigned char ch) { return std::isspace(ch) != 0; };
            line.erase(line.begin(),
                std::find_if(line.begin(), line.end(),
                    [&](char ch) { return !is_space((unsigned char)ch); }));
            line.erase(std::find_if(line.rbegin(), line.rend(),
                [&](char ch) { return !is_space((unsigned char)ch); }).base(),
                line.end());

            if (line.empty())
                continue;

            // Skip comments
            if (!line.empty() && (line[0] == '#' || line[0] == ';'))
                continue;

            std::istringstream iss(line);
            std::string gram;
            double      logp;

            if (!(iss >> gram >> logp))
            {
                // malformed line; skip
                continue;
            }

            // Normalize gram to uppercase
            std::transform(gram.begin(), gram.end(), gram.begin(),
                [](unsigned char ch) { return (char)std::toupper(ch); });

            if (gram.size() != 4)
            {
                // Not a quadgram; skip
                continue;
            }

            bool valid = true;
            for (char ch : gram)
            {
                if (ch < 'A' || ch > 'Z')
                {
                    valid = false;
                    break;
                }
            }
            if (!valid)
                continue;

            std::uint32_t code = pack_quadgram(gram);
            table[code] = logp;

            if (logp < min_logp)
                min_logp = logp;
        }

        if (!table.empty())
        {
            // Your note says lowest score is -24.000.
            // Use the min of (observed_min, -24.0) as precaution so we never raise the floor.
            if (std::isfinite(min_logp))
                floorLogPOut = std::min(min_logp, -24.0);
            else
                floorLogPOut = -24.0;
        }

        return table;
    }


    using TrigramTable = std::vector<double>; // size 26 * 26 * 26

    static inline std::uint32_t pack_trigram(const std::string& gram)
    {
        // gram must be 3 chars 'A'..'Z'
        int a = gram[0] - 'A';
        int b = gram[1] - 'A';
        int c = gram[2] - 'A';

        return ((std::uint32_t)a * 26u + (std::uint32_t)b) * 26u
            + (std::uint32_t)c;
    }

    static TrigramTable load_trigram_logcount_table(const std::string& filename)
    {
        std::ifstream in(filename);
        if (!in)
            throw std::runtime_error("Failed to open file: " + filename);

        constexpr std::size_t TRIGRAM_SPACE = 26u * 26u * 26u;
        TrigramTable table(TRIGRAM_SPACE, 0.0);

        std::string line;
        auto is_space = [](unsigned char ch) { return std::isspace(ch) != 0; };

        while (std::getline(in, line))
        {
            // Trim
            line.erase(
                line.begin(),
                std::find_if(line.begin(), line.end(),
                    [&](char ch) { return !is_space((unsigned char)ch); }));
            line.erase(
                std::find_if(line.rbegin(), line.rend(),
                    [&](char ch) { return !is_space((unsigned char)ch); }).base(),
                line.end());

            if (line.empty())
                continue;

            if (line[0] == '#' || line[0] == ';')
                continue;

            auto commaPos = line.find(',');
            if (commaPos == std::string::npos)
                continue; // malformed

            std::string gram = line.substr(0, commaPos);
            std::string countStr = line.substr(commaPos + 1);

            // Trim pieces
            gram.erase(
                gram.begin(),
                std::find_if(gram.begin(), gram.end(),
                    [&](char ch) { return !is_space((unsigned char)ch); }));
            gram.erase(
                std::find_if(gram.rbegin(), gram.rend(),
                    [&](char ch) { return !is_space((unsigned char)ch); }).base(),
                gram.end());

            countStr.erase(
                countStr.begin(),
                std::find_if(countStr.begin(), countStr.end(),
                    [&](char ch) { return !is_space((unsigned char)ch); }));
            countStr.erase(
                std::find_if(countStr.rbegin(), countStr.rend(),
                    [&](char ch) { return !is_space((unsigned char)ch); }).base(),
                countStr.end());

            if (gram.size() != 3 || countStr.empty())
                continue;

            // Normalize to uppercase
            std::transform(gram.begin(), gram.end(), gram.begin(),
                [](unsigned char ch) { return (char)std::toupper(ch); });

            bool valid = true;
            for (char ch : gram)
            {
                if (ch < 'A' || ch > 'Z')
                {
                    valid = false;
                    break;
                }
            }
            if (!valid)
                continue;

            std::uint64_t count = 0;
            try
            {
                count = std::stoull(countStr);
            }
            catch (...)
            {
                continue; // malformed count
            }

            if (count == 0)
                continue; // your data shouldn’t have this anyway

            std::uint32_t code = pack_trigram(gram);
            if (code >= TRIGRAM_SPACE)
                continue; // defensive

            table[code] = std::log(static_cast<double>(count)); // natural log
            // If you prefer log10, use std::log10 instead.
        }

        return table;
    }


};