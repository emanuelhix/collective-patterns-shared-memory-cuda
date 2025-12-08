CUDAVER_SCHOONER="CUDA/10.1.243-GCC-8.3.0"

todo:
	@grep -n -A 1 -B 2 STUDENT_TODO hw_code.cu || true
	@grep -n FIXME hw_code.cu || true
	@echo -e "\n\n"
	@echo -e "INSTRUCTIONS: Fix the code until every test passes."
	@echo -e "              You will also diagram the data movement in a separate pdf."
	@echo -e "run: make test"


# Use *-local if:
# 1. You have cuda installed on your machine and have a nvidia gpu.
# 2. You are running on the GPELs.
#
# Do not use this case on schooner. 
test-local: compile-local
	./run_hw.x | tee results.txt
	@grep --color PASS results.txt || true
	@grep --color FAIL results.txt || true
compile-local:
	nvcc hw_code.cu  -o ./run_hw.x

# FOR SCHOONER
# Use *-schooner if you are on schooner
compile-schooner:
	module load ${CUDAVER_SCHOONER}; \
	nvcc hw_code.cu  -o ./run_hw.x

run-schooner: compile-schooner
	sbatch hw.sbatch


package_for_submission: test
	tar czvf submit-to-canvas.tar.gz hw_code.cu results.txt

clean:
	rm -f *.x *~ *.o *.x results.txt  submit-to-canvas.tar.gz
