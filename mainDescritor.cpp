#include "funcoesArquivo.h"

int main(int argc, char *argv[]){

    string id, base, dir;
    int descritor, nColor, nRes, oNorm, quantMethod, oZero, totalpar;

    if (argc < 8){
        cout << "This program waits: <folder> <folder_descriptor> <descriptor> <image colors to quantize the image> ";
        cout << "<redimension factor> <normalization> <quantization method> <ACC distances | CCV threshold> ";
        cout << "<output id, if requested>\n";
        cout << " - Descriptors: 1 - BIC   2- GCH   3- CCV     4- Haralick    5- AutoCorrelograma (ACC)\n";
        cout << " - Colors: 8, 16, 32, 64 ou 256\n";
        cout << " - Redimension - positive, with max = 1 (1 = 100%)\n";
        cout << " - Normalization - 0 (without) 1 (between 0 and 1), 2 (0 a 255)\n";
        cout << " - Distances for ACC or threshold for CCV\n";
        exit(0);
    } 

    if (argc == 9){
        id = argv[8];
    }
    else{
        id = "";
    }

	base = argv[1];
	dir = argv[2];
	descritor = atoi(argv[3]);
	if((descritor < 1) || (descritor > 5)) {
	    cout << "Descriptor do not exists!!\n\n";
	    return -1;
	}
	nColor = atoi(argv[4]);
	nRes = atof(argv[5]);
	oNorm = atoi(argv[6]);
	quantMethod = atoi(argv[7]);
	oZero = 0;
	
	totalpar = (argc-8);
	int *params = new int[totalpar];
	if (descritor == 3) {
	    params[0] = atoi(argv[6]);
	}
	else if (descritor == 5){
        for (int i = 0; i < totalpar; i++){
            params[i] = atoi(argv[6+i]);
        }
	}
	
	if((nColor != 8) && (nColor != 16) && (nColor != 32) && (nColor != 64) && (nColor != 128) && (nColor != 256)){
	    cout << "Number of colors must be 8, 16, 32, 64, 128 or 256!!\n\n";
	    return -1;
	}
	if((nRes <= 0) || (nRes > 1)){
	    cout << "Redimension must be positive and less or equal to 1\n\n";
	    return -1;
	}
	if((oNorm < 0) || (oNorm > 2)){
	    cout << "Wrong normalization (0, 1 or 2)\n\n";
	    return -1;
	}
	
	descriptor(base.c_str(), dir.c_str(), descritor, nColor, nRes, oNorm, params, totalpar, oZero, quantMethod, id.c_str());
	
	return 1;
}