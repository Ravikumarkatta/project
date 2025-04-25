import pytest
from unittest.mock import Mock, patch

# Mock the lexicon functionality
class MockHebrewLexicon:
    def __init__(self):
        self.lexicon = {
            "בְּרֵאשִׁית": {
                "transliteration": "bereshit",
                "strongs_number": "H7225",
                "part_of_speech": "noun",
                "definition": "beginning, chief",
                "root": "ראש",
                "usage": "Used at the start of Genesis, 'In the beginning'",
                "related_words": ["ראש", "ראשון"],
                "semantic_domain": "time"
            },
            "אֱלֹהִים": {
                "transliteration": "elohim",
                "strongs_number": "H430",
                "part_of_speech": "noun",
                "definition": "God, gods, judges, angels",
                "root": "אלה",
                "usage": "Most often used as the generic term for God in the Hebrew Bible",
                "related_words": ["אל", "אלוה"],
                "semantic_domain": "deity"
            },
            "שָׁלוֹם": {
                "transliteration": "shalom",
                "strongs_number": "H7965",
                "part_of_speech": "noun",
                "definition": "peace, completeness, welfare, health",
                "root": "שלם",
                "usage": "Common greeting and state of wellbeing",
                "related_words": ["שלם", "שלומים"],
                "semantic_domain": "state"
            }
        }
    
    def lookup_word(self, hebrew_word):
        """Look up a Hebrew word in the lexicon."""
        return self.lexicon.get(hebrew_word, {"error": f"Word {hebrew_word} not found"})
    
    def lookup_by_strongs(self, strongs_number):
        """Look up a word by its Strong's number."""
        for word, data in self.lexicon.items():
            if data.get("strongs_number") == strongs_number:
                return {word: data}
        return {"error": f"Strong's number {strongs_number} not found"}
    
    def get_word_family(self, root):
        """Get all words sharing the same root."""
        family = {}
        for word, data in self.lexicon.items():
            if data.get("root") == root:
                family[word] = data
        
        if not family:
            return {"error": f"No words found for root {root}"}
        
        return family


class MockGreekLexicon:
    def __init__(self):
        self.lexicon = {
            "λόγος": {
                "transliteration": "logos",
                "strongs_number": "G3056",
                "part_of_speech": "noun",
                "definition": "word, speech, account, reason",
                "root": "λεγω",
                "usage": "Used in John 1:1, 'In the beginning was the Word'",
                "related_words": ["λεγω", "λογιζομαι"],
                "semantic_domain": "communication"
            },
            "ἀγάπη": {
                "transliteration": "agape",
                "strongs_number": "G26",
                "part_of_speech": "noun",
                "definition": "love, goodwill, benevolence",
                "root": "αγαπαω",
                "usage": "Selfless, sacrificial love emphasized in the New Testament",
                "related_words": ["αγαπαω", "αγαπητος"],
                "semantic_domain": "emotion"
            },
            "χάρις": {
                "transliteration": "charis",
                "strongs_number": "G5485",
                "part_of_speech": "noun",
                "definition": "grace, favor, kindness",
                "root": "χαιρω",
                "usage": "Divine grace often emphasized in Paul's epistles",
                "related_words": ["χαιρω", "χαρα"],
                "semantic_domain": "attribute"
            }
        }
    
    def lookup_word(self, greek_word):
        """Look up a Greek word in the lexicon."""
        return self.lexicon.get(greek_word, {"error": f"Word {greek_word} not found"})
    
    def lookup_by_strongs(self, strongs_number):
        """Look up a word by its Strong's number."""
        for word, data in self.lexicon.items():
            if data.get("strongs_number") == strongs_number:
                return {word: data}
        return {"error": f"Strong's number {strongs_number} not found"}
    
    def get_word_family(self, root):
        """Get all words sharing the same root."""
        family = {}
        for word, data in self.lexicon.items():
            if data.get("root") == root:
                family[word] = data
        
        if not family:
            return {"error": f"No words found for root {root}"}
        
        return family


class MockConcordance:
    def __init__(self, hebrew_lexicon, greek_lexicon):
        self.hebrew_lexicon = hebrew_lexicon
        self.greek_lexicon = greek_lexicon
        
        # Mock verse references for specific words
        self.word_references = {
            "בְּרֵאשִׁית": ["Gen 1:1", "Jer 26:1", "Prov 8:22"],
            "אֱלֹהִים": ["Gen 1:1", "Gen 1:2", "Exod 20:1", "Psa 82:1"],
            "שָׁלוֹם": ["Num 6:26", "Psa 29:11", "Isa 9:6"],
            "λόγος": ["John 1:1", "John 1:14", "1 John 1:1", "Rev 19:13"],
            "ἀγάπη": ["1 Cor 13:1", "1 Cor 13:13", "1 John 4:8", "1 John 4:16"],
            "χάρις": ["John 1:17", "Rom 3:24", "Eph 2:8", "Tit 2:11"]
        }
    
    def get_occurrences(self, word):
        """Get all verse references where a word occurs."""
        if word in self.word_references:
            return self.word_references[word]
        
        # Check if it's a Strong's number
        if word.startswith("H") or word.startswith("G"):
            if word.startswith("H"):
                result = self.hebrew_lexicon.lookup_by_strongs(word)
            else:
                result = self.greek_lexicon.lookup_by_strongs(word)
            
            if "error" not in result:
                actual_word = list(result.keys())[0]
                return self.word_references.get(actual_word, [])
        
        return []
    
    def get_verse_words(self, verse_ref):
        """Get all significant words in a verse with their lexical data."""
        # Mock implementation for testing
        verse_words = {
            "Gen 1:1": ["בְּרֵאשִׁית", "אֱלֹהִים"],
            "John 1:1": ["λόγος"]
        }
        
        words_data = {}
        for word in verse_words.get(verse_ref, []):
            if word in self.hebrew_lexicon.lexicon:
                words_data[word] = self.hebrew_lexicon.lookup_word(word)
            elif word in self.greek_lexicon.lexicon:
                words_data[word] = self.greek_lexicon.lookup_word(word)
        
        return words_data
