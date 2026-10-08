"""
Homework 2 - 10 RegEx Problems

Problems:
    1. Extract email addresses
    2. Validate phone numbers (XXX-XXX-XXXX)
    3. Extract dates (DD/MM/YYYY or DD-MM-YYYY)
    4. Find repeated words
    5. Extract hashtags
    6. Validate password strength
    7. Extract URLs
    8. Replace multiple spaces with a single space
    9. Extract quoted text
    10. Validate IP addresses

Only Python's built-in `re` module is used.
"""

import re


# ---------------------------------------------------------------------------
# Problem 1: Extract Email Addresses
# ---------------------------------------------------------------------------
EMAIL_TEXT = (
    "Contact us at support@example.com or sales@company.org for assistance.\n"
    "For personal inquiries, email john.doe123@university.edu."
)

# local part: letters, digits and . _ % + -
# domain: one or more labels followed by a dot, then a TLD of 2+ letters.
# The TLD must be letters only, so the sentence-ending "." is not included.
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}\b")


def extract_emails(text):
    """Return every email address found in the text."""
    return EMAIL_PATTERN.findall(text)


# ---------------------------------------------------------------------------
# Problem 2: Validate Phone Numbers
# ---------------------------------------------------------------------------
PHONE_TEXT = (
    "Valid: 123-456-7890, 987-654-3210\n"
    "Invalid: 12-345-67890, 1234567890, 123-45-6789"
)

# Exactly 3-3-4 digits. The lookarounds make sure the number is not part of a
# longer digit/dash sequence (e.g. "345-6789" inside "12-345-67890").
PHONE_PATTERN = re.compile(r"(?<![\d-])\d{3}-\d{3}-\d{4}(?![\d-])")


def find_valid_phones(text):
    """Return phone numbers that follow the XXX-XXX-XXXX format."""
    return PHONE_PATTERN.findall(text)


# ---------------------------------------------------------------------------
# Problem 3: Extract Dates
# ---------------------------------------------------------------------------
DATE_TEXT = "Important dates: 25/12/2023, 01-01-2024, 31/05/2023, and 15-10-2024."

# DD: 01-31, MM: 01-12, YYYY: four digits.
# The separator is captured and reused with \1 so mixed forms such as
# "25/12-2023" are rejected.
DATE_PATTERN = re.compile(
    r"\b(?:0[1-9]|[12]\d|3[01])([/-])(?:0[1-9]|1[0-2])\1\d{4}\b"
)


def extract_dates(text):
    """Return all dates written as DD/MM/YYYY or DD-MM-YYYY."""
    return [match.group(0) for match in DATE_PATTERN.finditer(text)]


# ---------------------------------------------------------------------------
# Problem 4: Find Repeated Words
# ---------------------------------------------------------------------------
REPEATED_TEXT = "The the quick brown fox jumps over the the lazy dog."

# (\w+) captures a word, \s+ the gap, \1 the same word again.
# IGNORECASE lets "The the" count as a repetition.
REPEATED_PATTERN = re.compile(r"\b(\w+)\s+\1\b", re.IGNORECASE)


def find_repeated_words(text):
    """Return every pair of consecutive repeated words (lowercased)."""
    return [match.group(0).lower() for match in REPEATED_PATTERN.finditer(text)]


# ---------------------------------------------------------------------------
# Problem 5: Extract Hashtags
# ---------------------------------------------------------------------------
HASHTAG_TEXT = "Check out our new products: #Sale2024, #NewArrival, and #Discounts!"

# "#" followed by one or more word characters (letters, digits, underscore).
HASHTAG_PATTERN = re.compile(r"#\w+")


def extract_hashtags(text):
    """Return all hashtags in the text."""
    return HASHTAG_PATTERN.findall(text)


# ---------------------------------------------------------------------------
# Problem 6: Validate Password Strength
# ---------------------------------------------------------------------------
PASSWORDS = ["Password123", "Secure456", "weak", "password", "Password"]

# Each lookahead checks one rule without consuming characters:
#   (?=.*[a-z])  at least one lowercase letter
#   (?=.*[A-Z])  at least one uppercase letter
#   (?=.*\d)     at least one digit
#   .{8,}        at least 8 characters in total
PASSWORD_PATTERN = re.compile(r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d).{8,}$")


def find_valid_passwords(passwords):
    """Return the passwords that satisfy all strength rules."""
    return [password for password in passwords if PASSWORD_PATTERN.fullmatch(password)]


# ---------------------------------------------------------------------------
# Problem 7: Extract URLs
# ---------------------------------------------------------------------------
URL_TEXT = (
    "Visit our website at https://www.example.com or check out "
    "http://blog.example.org for updates."
)

# http or https, then domain labels and a TLD, then an optional path.
# The path may not end with punctuation, so a trailing "." or "," is dropped.
URL_PATTERN = re.compile(
    r"https?://(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}(?::\d+)?(?:/[^\s]*[^\s.,!?;:)])?/?"
)


def extract_urls(text):
    """Return all http/https URLs in the text."""
    return URL_PATTERN.findall(text)


# ---------------------------------------------------------------------------
# Problem 8: Replace Multiple Spaces with a Single Space
# ---------------------------------------------------------------------------
SPACES_TEXT = "This    text   has    multiple     spaces."

# Two or more spaces in a row -> one space.
SPACES_PATTERN = re.compile(r" {2,}")


def collapse_spaces(text):
    """Replace every run of multiple spaces with a single space."""
    return SPACES_PATTERN.sub(" ", text)


# ---------------------------------------------------------------------------
# Problem 9: Extract Quoted Text
# ---------------------------------------------------------------------------
QUOTED_TEXT = 'He said, "Hello, world!" and she replied, "Hi there!"'

# Opening quote, then any characters that are not a quote (captured), then the
# closing quote. [^"]* keeps each match inside a single pair of quotes.
QUOTED_PATTERN = re.compile(r'"([^"]*)"')


def extract_quoted(text):
    """Return the contents of every double-quoted string."""
    return QUOTED_PATTERN.findall(text)


# ---------------------------------------------------------------------------
# Problem 10: Validate IP Addresses
# ---------------------------------------------------------------------------
IP_TEXT = (
    "Valid: 192.168.1.1, 10.0.0.255\n"
    "Invalid: 256.1.2.3, 192.168.01.1, 192.168.1"
)

# One octet: 0-255 without leading zeros.
#   25[0-5]      250-255
#   2[0-4]\d     200-249
#   1\d\d        100-199
#   [1-9]?\d     0-99 (no leading zero, so "01" is rejected)
OCTET = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"

# Four octets separated by dots. The lookarounds stop partial matches such as
# "56.1.2.3" inside "256.1.2.3" or "192.168.1" being part of a longer address.
IP_PATTERN = re.compile(rf"(?<![\d.]){OCTET}(?:\.{OCTET}){{3}}(?!\.?\d)")


def find_valid_ips(text):
    """Return the valid IPv4 addresses found in the text."""
    return IP_PATTERN.findall(text)


# ---------------------------------------------------------------------------
# Run all problems
# ---------------------------------------------------------------------------
def print_lines(title, items):
    print(title)
    for item in items:
        print(item)
    print()


def main():
    print_lines("Problem 1: Extract Email Addresses", extract_emails(EMAIL_TEXT))

    print("Problem 2: Validate Phone Numbers")
    print_lines("Valid phone numbers:", find_valid_phones(PHONE_TEXT))

    print_lines("Problem 3: Extract Dates", extract_dates(DATE_TEXT))

    print("Problem 4: Find Repeated Words")
    print_lines("Repeated words:", find_repeated_words(REPEATED_TEXT))

    print_lines("Problem 5: Extract Hashtags", extract_hashtags(HASHTAG_TEXT))

    print("Problem 6: Validate Password Strength")
    print_lines("Valid passwords:", find_valid_passwords(PASSWORDS))

    print_lines("Problem 7: Extract URLs", extract_urls(URL_TEXT))

    print_lines("Problem 8: Replace Multiple Spaces with a Single Space",
                [collapse_spaces(SPACES_TEXT)])

    print_lines("Problem 9: Extract Quoted Text", extract_quoted(QUOTED_TEXT))

    print("Problem 10: Validate IP Addresses")
    print_lines("Valid IP addresses:", find_valid_ips(IP_TEXT))


if __name__ == "__main__":
    main()
