
# pandoc and pypandoc need to be installed to generate the documentation.
docs:
	# Copy Sonnet logo into the right location.
	mkdir -p docs/images && cp images/sonnet_logo.png docs/images/sonnet_logo.png
	# Convert our README.md into an rst file.
	pandoc -f markdown -o docs/README.rst docs/README.md
	# Create automated API docs.
	sphinx-apidoc -o docs/api sonnet/ 'sonnet/python/ops/*_test*'
	# Build Sphinx documentation.
	sphinx-build docs/ docs/_build

clean:
	rm -r docs/_build
	rm -r docs/api
	rm -r docs/images
	rm docs/README.rst

.PHONY: docs clean
