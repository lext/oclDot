clDot:
	gcc -std=gnu99 -c cl-helper.c -O3
	gcc -std=gnu99 oclDot.c -o oclDot cl-helper.o -lm -lOpenCL -Wall -O3
	./oclDot
clean:
	rm *.o oclDot
	

