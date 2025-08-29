from pathlib import Path

# parse a comma-separated string into a list
def parse_comma_separated(text, as_type=str):
    items = [items.strip() for item in text.split(',') if item.strip()]
    if as_type is not str:
        try:
            items = [as_type(item) for item in items]
        except Exception:
            pass
    return items

# create folder if it doesn't exist
def ensure_folder(path):
    Path(path).mkdir(parents=True, exist_ok=True)

# return the filename from a path
def get_filename(path):
    return Path(path).name

# show QMessageBox error
def show_error(parent, message):
    from PySide6.QtWidgets import QMessageBox
    QMessageBox.critical(parent, 'Error', message)

# show QMessageBox information
def show_info(parent, message):
    from PySide6.QtWidgets import QMessageBox
    QMessageBox.information(parent, 'Info', message)

# format a float to a given precision
def format_field(val, precision=2):
    try:
        return f'{float(val):.{precision}f}'
    except Exception:
        return str(val)