# Makefile

# Define the test target
test:
	@echo "Running tests..."
	@pushd ./build && cmake .. && make -j && popd
	@python3.10 test.py

# Define a clean target to remove build files
clean:
	@echo "Cleaning up..."
	@rm -rf ./build/*