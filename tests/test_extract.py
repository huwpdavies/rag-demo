"""Tests for the PDF text cleaning rules (on hand-written layout text, no PDF needed)."""

from rag_demo.extract import clean_page_text


def test_tabs_and_repeated_spaces_collapse():
    assert clean_page_text("salt\tfat   acid\t\theat") == "salt fat acid heat"


def test_wrapped_lines_join_into_one_paragraph():
    assert clean_page_text("Season the water\nuntil it tastes\nlike the sea.") == "Season the water until it tastes like the sea."


def test_line_end_hyphen_is_rejoined_and_kept():
    # In this book, line-end hyphens are real compound words.
    assert clean_page_text("Use ice-\ncold water.") == "Use ice-cold water."


def test_indented_line_after_a_sentence_starts_a_new_paragraph():
    raw = "The first paragraph ends here.\n     The second one starts here."
    assert clean_page_text(raw) == "The first paragraph ends here.\n\nThe second one starts here."


def test_hanging_indent_mid_sentence_does_not_split_the_paragraph():
    raw = "     Add enough oil to go two-\n     thirds of the way up the sides."
    assert clean_page_text(raw) == "Add enough oil to go two-thirds of the way up the sides."


def test_bullets_start_new_paragraphs():
    raw = "Variations\n• Use pork instead\n• Or chicken"
    assert clean_page_text(raw) == "Variations\n\n• Use pork instead\n\n• Or chicken"


def test_blank_page_gives_empty_text():
    assert clean_page_text("   \n\n  ") == ""
