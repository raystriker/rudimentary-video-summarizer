from misc_utils import sanitize_filename


class TestSanitizeFilename:
    def test_removes_special_characters(self):
        assert sanitize_filename('file*name?.txt') == "filename.txt"

    def test_replaces_spaces_with_underscores(self):
        assert sanitize_filename("my cool video.mp3") == "my_cool_video.mp3"

    def test_handles_multiple_spaces(self):
        assert sanitize_filename("lots   of   spaces") == "lots_of_spaces"

    def test_removes_all_invalid_chars(self):
        assert sanitize_filename('a\\b/c*d?e"f<g>h|i') == "abcdefghi"

    def test_preserves_valid_filename(self):
        assert sanitize_filename("already_valid.txt") == "already_valid.txt"

    def test_empty_string(self):
        assert sanitize_filename("") == ""

    def test_combined_special_chars_and_spaces(self):
        assert sanitize_filename('My Video: "Part 1" | Finale') == "My_Video_Part_1_Finale"
