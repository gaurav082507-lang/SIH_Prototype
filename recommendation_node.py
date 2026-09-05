# =========================================================
# RETRY WRAPPER FOR TRANSIENT GEMINI ERRORS
# =========================================================

def _invoke_with_retry(llm, messages, max_attempts=4, base_delay=1.0):
    """
    Retries on transient 503 UNAVAILABLE / 429 RESOURCE_EXHAUSTED
    errors with exponential backoff + jitter.
    """

    last_error = None

    for attempt in range(1, max_attempts + 1):

        try:
            return llm.invoke(messages)

        except (ServiceUnavailable, ResourceExhausted) as error:
            last_error = error

        except Exception as error:
            # Some SDK versions raise a generic exception whose
            # string contains "503" / "UNAVAILABLE" instead of the
            # typed google.api_core exceptions above.
            error_text = str(error)
            if "503" in error_text or "UNAVAILABLE" in error_text or "429" in error_text:
                last_error = error
            else:
                raise

        if attempt < max_attempts:
            delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
            print(
                f"[recommendation_node] transient error on attempt "
                f"{attempt}/{max_attempts}, retrying in {delay:.1f}s: "
                f"{last_error}"
            )
            time.sleep(delay)

    raise last_error
