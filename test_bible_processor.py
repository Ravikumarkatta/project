import unittest
import tempfile
import os
import json
from kjv_preprocessor import BibleProcessor, BIBLE_BOOKS

class TestBibleProcessor(unittest.TestCase):
    
    def setUp(self):
        self.processor = BibleProcessor(include_apocrypha=False, fix_line_breaks=True)
        
    def test_remove_gutenberg_wrappers(self):
        text = """*** START OF THE PROJECT GUTENBERG EBOOK THE BIBLE ***
        
        The Bible
        
        Genesis
        
        *** END OF THE PROJECT GUTENBERG EBOOK THE BIBLE ***"""
        
        processed = self.processor.remove_gutenberg_wrappers(text)
        self.assertEqual(processed, "The Bible\n        \n        Genesis")
        
    def test_normalize_book_name(self):
        # Test direct matches
        self.assertEqual(self.processor.normalize_book_name("Genesis"), "Genesis")
        self.assertEqual(self.processor.normalize_book_name("Revelation"), "Revelation")
        
        # Test aliases
        self.assertEqual(self.processor.normalize_book_name("Psalm"), "Psalms")
        self.assertEqual(self.processor.normalize_book_name("The Revelation"), "Revelation")
        
        # Test numbered books
        self.assertEqual(self.processor.normalize_book_name("1 John"), "1 John")
        self.assertEqual(self.processor.normalize_book_name("First John"), "1 John")
        self.assertEqual(self.processor.normalize_book_name("1 John"), "1 John")
        
        # Test special cases
        self.assertEqual(self.processor.normalize_book_name("Samuel"), "1 Samuel")
        
        # Test unknown books
        self.assertIsNone(self.processor.normalize_book_name("NotABook"))
        
    def test_identify_verse_reference(self):
        # Test standard chapter:verse format
        self.assertEqual(
            self.processor.identify_verse_reference("1:1 In the beginning"),
            (1, 1, "In the beginning")
        )
        
        # Test just verse number
        self.processor.current_chapter = 2
        self.assertEqual(
            self.processor.identify_verse_reference("3 And God said"),
            (2, 3, "And God said")
        )
        
        # Test verse with brackets
        self.assertEqual(
            self.processor.identify_verse_reference("[4] Let there be light"),
            (2, 4, "Let there be light")
        )
        
        # Test non-verse line
        self.assertIsNone(self.processor.identify_verse_reference("This is not a verse"))
        
    def test_process_bible_text(self):
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as input_file:
            input_file.write("""*** START OF THE PROJECT GUTENBERG EBOOK THE BIBLE ***
            
            The Book of Genesis
            
            Chapter 1
            
            1:1 In the beginning God created the heaven and the earth.
            1:2 And the earth was without form, and void.
            
            *** END OF THE PROJECT GUTENBERG EBOOK THE BIBLE ***""")
            input_path = input_file.name
            
        with tempfile.NamedTemporaryFile(delete=False) as output_file:
            output_path = output_file.name
            
        try:
            report = self.processor.process_bible_text(input_path, output_path, "json", True)
            
            # Check report contents
            self.assertEqual(report["books_processed"], 1)
            self.assertEqual(report["verses_processed"], 2)
            
            # Check output file structure
            with open(output_path, 'r') as f:
                output_data = json.load(f)
                
            self.assertIn("Genesis", output_data)
            self.assertIn(1, output_data["Genesis"])
            self.assertEqual(output_data["Genesis"][1][1], "In the beginning God created the heaven and the earth.")
            self.assertEqual(output_data["Genesis"][1][2], "And the earth was without form, and void.")
                
        finally:
            os.unlink(input_path)
            os.unlink(output_path)
            
    def test_validation(self):
        # Create a valid but tiny Bible structure
        self.processor.parsed_bible = {
            "Genesis": {
                1: {
                    1: "In the beginning God created the heaven and the earth."
                }
            }
        }
        self.processor.stats.books_found.add("Genesis")
        self.processor.stats.add_chapter("Genesis", 1)
        self.processor.stats.add_verse("Genesis", 1, 1)
        
        report = self.processor.validate_bible_structure()
        
        # It should warn about low verse count
        self.assertFalse(report["valid"])
        self.assertTrue(any("Low verse count" in warning for warning in report["warnings"]))
        
        # It should warn about missing books
        self.assertTrue(any("Missing books" in warning for warning in report["warnings"]))
        

if __name__ == "__main__":
    unittest.main()