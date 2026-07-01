#!/bin/bash

# Build jupyter book
jupyter-book build .

# Publish to GitHub Pages
ghp-import -n -p -f _build/html