"""This is the scene generation pipe library.
"""


import logging
# This prevents "No handler found" warnings if the user doesn't configure logging
logging.getLogger(__name__).addHandler(logging.NullHandler())