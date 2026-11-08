"""Small text-output helper for native statistical test results."""


def write_native_result(output, label, result):
    output.write(label + "\n")
    output.write(repr(result) + "\n\n")


def anonymize_summary(text):
    """Withhold run timestamps while preserving every statistical field."""
    import re

    pattern = r"(?m)^(Date:|Time:)(.*?)(?=\s{2,}(?:Deviance:|Pearson chi2:))"
    return re.sub(
        pattern,
        lambda match: (match.group(1) + " [withheld for review]").ljust(
            len(match.group(0))
        ),
        text,
    )
