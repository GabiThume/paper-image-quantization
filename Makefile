OPENCV = -I /usr/include/opencv
OPENCVCONF = `pkg-config opencv --libs`
FIND=find -name
RM=rm -rf

all: clean descritores funcoesAux funcoesArquivo merge_datasets classifier dimensionReduction
	@g++ -Wall descritores.o funcoesAux.o funcoesArquivo.o mainDescritor.cpp -o mainDescritor $(OPENCV) $(OPENCVCONF)
	
descritores:
	@g++ -Wall -c -g descritores.cpp $(OPENCV)

funcoesAux:
	@g++ -Wall -c -g funcoesAux.cpp $(OPENCV)
	
funcoesArquivo:
	@g++ -Wall -c -g funcoesArquivo.cpp $(OPENCV)

dimensionReduction: dimensionReduction.cpp
	@g++ -Wall -o dimensionReduction descritores.o funcoesAux.o funcoesArquivo.o classifier.o dimensionReduction.cpp $(OPENCV) $(OPENCVCONF)
	
merge_datasets: mergeDataSets.cpp	
	@g++ -Wall -o mergeDataSets mergeDataSets.cpp

classifier:
	@g++ -Wall -c -g classifier.cpp $(OPENCV)

clean:
	$(FIND) "*~" | xargs $(RM)
	$(RM) teste mainDescritor dimensionReduction mergeDataSets

generate:
	./runAllDescriptors.sh

run:
	./dimensionReduction

