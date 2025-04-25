import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class VerseReference:
    book: str
    chapter: int
    verse: Optional[int] = None
    end_verse: Optional[int] = None
    end_chapter: Optional[int] = None


class StandaloneVerseDetector:
    def __init__(self):
        self.book_names = {
            "genesis": "Genesis",
            "exodus": "Exodus",
            "matthew": "Matthew",
            "mark": "Mark",
            "luke": "Luke",
            "john": "John",
            "psalms": "Psalms",
            "psalm": "Psalms",
        }

        self.patterns = [
            # Cross-chapter range (e.g. Psalm 1:1-2:3)
            r"\b({})\s+(\d+)[:.](\d+)-(\d+)[:.](\d+)\b".format(self._book_pattern()),
            # Same chapter range (e.g. Matthew 5:3-12)
            r"\b({})\s+(\d+)[:.](\d+)-(\d+)\b".format(self._book_pattern()),
            # Single verse (e.g. John 3:16)
            r"\b({})\s+(\d+)[:.](\d+)\b".format(self._book_pattern()),
            # Whole chapter (e.g. Genesis 1)
            r"\b({})\s+(\d+)\b(?![:.])".format(self._book_pattern()),
        ]

        self.compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.patterns]

    def _book_pattern(self):
        return "|".join(map(re.escape, self.book_names.keys()))

    def detect_references(self, text: str) -> List[VerseReference]:
        references = []
        seen_spans = set()

        # First find all matches and their spans
        all_matches = []
        for pattern in self.compiled_patterns:
            for match in pattern.finditer(text):
                all_matches.append((match.span(), match.groups()))

        # Process matches from longest to shortest to prefer more specific patterns
        all_matches.sort(key=lambda x: x[0][1] - x[0][0], reverse=True)

        for span, groups in all_matches:
            # Skip if any part of this span is already covered
            if any(
                s[0] <= span[0] < s[1] or s[0] < span[1] <= s[1] for s in seen_spans
            ):
                continue

            seen_spans.add(span)
            book = self.book_names[groups[0].lower()]

            if len(groups) == 2:  # Whole chapter
                references.append(VerseReference(book=book, chapter=int(groups[1])))
            elif len(groups) == 3:  # Single verse
                references.append(
                    VerseReference(
                        book=book, chapter=int(groups[1]), verse=int(groups[2])
                    )
                )
            elif len(groups) == 4:  # Same chapter range
                references.append(
                    VerseReference(
                        book=book,
                        chapter=int(groups[1]),
                        verse=int(groups[2]),
                        end_verse=int(groups[3]),
                    )
                )
            elif len(groups) == 5:  # Cross-chapter range
                references.append(
                    VerseReference(
                        book=book,
                        chapter=int(groups[1]),
                        verse=int(groups[2]),
                        end_chapter=int(groups[3]),
                        end_verse=int(groups[4]),
                    )
                )

        return references
