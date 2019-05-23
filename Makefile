all:
	mkdir -p Release
	clang++ DeepLearningCpp.cpp -o ./Release/DeepLearningCpp.exe -O3 -std=c++11 -Wc++11-extensions -DFAST_MATRIX_MULTIPLY
naive:
	mkdir -p Release
	clang++ DeepLearningCpp.cpp -o ./Release/DeepLearningCpp.exe -O3 -std=c++11 -Wc++11-extensions
clean:
	mkdir -p Release
	rm -f ./Release/*.o ./Release/*.exe
