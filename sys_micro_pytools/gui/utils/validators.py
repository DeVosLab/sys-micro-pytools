import os

# return True if file exists and is a file
def file_exists(path):
    return os.path.isfile(path)

# return True if folder exists
def folder_exists(path):
    return os.path.isdir(path)

# return True if s can be converted to float
def is_float(s):
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False
    
# return True if s can be converted to integer
def is_int(s):
    try:
        int(s)
        return True
    except (TypeError, ValueError):
        return False

 # return True if s is not empty 
def not_empty(s):
    return bool(s and s.strip())

# fields: dictionary of {field_name: value}
# returns: (bool, message)
def validate_required_fields(fields):
    for name, value in fields.items():
        if not not_empty(value):
            return False, f"Field '{name}' is required."
    return True, ""