import enum


@enum.unique
class CheckStatus(enum.IntEnum):
    STATUS_UNAVAIL = -3  # Check is not implemented yet
    STATUS_UNKNOWN = -2  # Default value when check is not done
    STATUS_ERROR = -1  # Prerequisite not satisfied (e.g., no foreground object, eyes not detected) when performing check
    STATUS_FAIL = 0  # Check fail (not because of internal error)
    STATUS_PASS = 1  # Check pass
