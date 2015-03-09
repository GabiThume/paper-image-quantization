/**
 * 
 *	Authors:	
 *			Luciana Calixta Escobar
 *			Gabriela Thumé
 *	Universidade de São Paulo / ICMC / 2014
 **/

#include "descritores.h"
#include "funcoesAux.h"


void find_neighbor(Mat & img, queue<Pixel> & pixels, int * visited, long int & tam_reg){

	int height = img.rows;
	int width = img.cols;
	Pixel pix = pixels.front();
	pixels.pop();
	int i = pix.i;
	int j = pix.j;
	uchar pix_color = pix.color;
	uchar img_color;
	int s, t;
	s = i - 1;
	t = j;
	
	if (s >= 0 && s < height && t >= 0 && t < width) {
	    img_color = img.at<uchar>(s,t);
	    if (visited[s*width + t] == 0 && img_color == pix_color) {
		pix.i = s;
		pix.j = t;
		pixels.push (pix);
		visited[s*width + t] = 1;
		tam_reg++;
	    }
	}

	s = i;
	t = j + 1;
	
	if (s >= 0 && s < height && t >= 0 && t < width) {
	    img_color = img.at<uchar>(s,t);
	    if (visited[s*width + t] == 0 && img_color == pix_color) {
		pix.i = s;
		pix.j = t;
		pixels.push (pix);
		visited[s*width + t] = 1;
		tam_reg++;
	    }
	}

	s = i + 1;
	t = j;
	if (s >= 0 && s < height && t >= 0 && t < width) {
	    img_color = img.at<uchar>(s,t);
	    if (visited[s*width + t] == 0 && img_color == pix_color) {
		pix.i = s;
		pix.j = t;
		pixels.push (pix);
		visited[s*width + t] = 1;
		tam_reg++;
	    }
	}

	s = i;
	t = j - 1;
	if (s >= 0 && s < height && t >= 0 && t < width) {
	    img_color = img.at<uchar>(s,t);
	    if (visited[s*width + t] == 0 && img_color == pix_color) {
		pix.i = s;
		pix.j = t;
		pixels.push (pix);
		visited[s*width + t] = 1;
		tam_reg++;
	    }
	}
	
}

void CCV(Mat & img, Mat &features, int nColor, int oNorm, int threshold){

	int i, j;
	long int *descriptor = new long int[nColor*2];

	for(i = 0; i < nColor*2; i++){
		descriptor[i] = 0;
	}

	Mat img_quant(img.size(), CV_8UC1);
	int height = img_quant.rows;
	int width = img_quant.cols;

	if (img.channels() == 1){
		double min, max;
		Point maxLoc, minLoc;
		minMaxLoc(img, &min, &max, &minLoc, &maxLoc);
		double stretch = ((double)((nColor-1)) / (max - min ));
		img_quant = img - min;
		img_quant = img_quant * stretch;
	}
	else {
	    QuantizationMSB(img, img_quant, nColor);
	}

	int *pxl_visited = new int[height*width]();
	long int tam_reg;

	queue<Pixel> pixels;
	Pixel pix;

	for (i = 0; i < height; i++){
		for (j = 0; j < width; j++){
			if (!pxl_visited[i*width + j]){
				pix.i = i; 
				pix.j = j;
				pix.color = img_quant.at<uchar>(i, j);

				pixels.push (pix);
				pxl_visited[i*width + j] = 1;
				tam_reg = 1;
		
				while (!pixels.empty())
					find_neighbor(img_quant, pixels, pxl_visited, tam_reg);

				if(tam_reg >= threshold)
					descriptor[pix.color*2 + 0] += tam_reg;
				else
					descriptor[pix.color*2 + 1] += tam_reg;
			}
		}
	}

	if (oNorm == 0) {
		for (i = 0; i < nColor*2 ; i++){
			features.at<float>(0,i) = (float)descriptor[i];
		}
	}
	else {
	    float *norm = new float[nColor*2];
	    if (oNorm == 1) 
			NormalizeHist(descriptor, norm, 2*nColor, 1);
	    else if (oNorm == 2) 
			NormalizeHist(descriptor, norm, 2*nColor, 255);
	    for (i = 0; i < nColor*2 ; i++){
			features.at<float>(0,i) = norm[i];
	    }
	    delete[] norm;
	}
	
	delete[] pxl_visited;
	delete[] descriptor;
}

void GCH(Mat &I, Mat &features, int nColor, int oNorm){
	Mat Q(I.size(), CV_8U, 1);
	
	if (I.channels() == 1) {
	      double min, max;
	      Point maxLoc, minLoc;
	      minMaxLoc(I, &min, &max, &minLoc, &maxLoc);
	      double stretch = ((double)((nColor-1)) / (max - min ));
	      Q = I - min;
	      Q = Q * stretch;
	}
	else {
	    QuantizationMSB(I, Q, nColor);
	}
	
	int i;
	long int *hist = new long int[nColor];
	
	for (i = 0; i < nColor; i++){
		hist[i] = 0;
	}
	
	MatIterator_<uchar> it, end;
	end = Q.end<uchar>();
	
	for (it = Q.begin<uchar>(); it != end; ++it){
		hist[(*it)]++;
	}
	
	if (oNorm == 0) {
		for (i = 0; i < nColor ; i++){
			features.at<float>(0,i) = (float)hist[i];
		}
	}
	else {
	    float *norm = new float[nColor];
	    if (oNorm == 1) 
			NormalizeHist(hist, norm, nColor, 1);
	    else if (oNorm == 2) 
			NormalizeHist(hist, norm, nColor, 255);
	    
	    for (i = 0; i < nColor ; i++){
			features.at<float>(0,i) = norm[i];
	    }
	    delete[] norm;
	}
	
	delete[] hist;
}

void BIC(Mat &I, Mat &features, int nColor, int oNorm) 
{
	Size imgSize = I.size();
	int height = imgSize.height;
	int width = imgSize.width;
	
	Mat Q(imgSize, CV_8U, 1);
	
	if (I.channels() == 1) {
	      double min, max;
	      Point maxLoc, minLoc;
	      minMaxLoc(I, &min, &max, &minLoc, &maxLoc);
	      double stretch = ((double)((nColor-1)) / (max - min ));
	      Q = I - min;
	      Q = Q * stretch;
	}
	else {
	      QuantizationMSB(I, Q, nColor);
	}
	
	int i, j;
	long int *hist = new long int[2*nColor];

	for (i = 0; i < 2*nColor; i++){
		hist[i] = 0;
	}
	
	for (i = 0; i < height; i++){
		for (j = 0; j < width; j++){
			uchar aux = Q.at<uchar>(i,j);
			if (i > 0 && j > 0 && j < width-1 && i < height-1){
				if ((Q.at<uchar>(i,j-1) == aux) && 
					(Q.at<uchar>(i,j+1) == aux) && 
					(Q.at<uchar>(i-1,j) == aux) && 
					(Q.at<uchar>(i+1,j) == aux)) {
					hist[aux]++;
				}
				else 
					hist[aux+nColor]++;
			}
			else
				hist[aux+nColor]++;
			  
		}
	}
	
	if (oNorm == 0) {
	  for (i = 0; i < nColor*2 ; i++){
		features.at<float>(0,i) = (float)hist[i];
	  }
	}
	else {
	    float *norm = new float[nColor*2];
	    if (oNorm == 1) 
			NormalizeHist(hist, norm, nColor*2, 1);
	    else if (oNorm == 2) 
			NormalizeHist(hist, norm, nColor*2, 1024);
	    
	    for (i = 0; i < nColor*2 ; i++){
			features.at<float>(0,i) = norm[i];
	    }
	    delete[] norm;
	}

	delete[] hist;
}

void CoocurrenceMatrix(Mat &I, double **Cm, int nColor, int dX, int dY){
	int i,j;
	
	Mat Q(I.size(), CV_8U, 1);
	if (I.channels() == 1){
	      double min, max;
	      Point maxLoc, minLoc;
	      minMaxLoc(I, &min, &max, &minLoc, &maxLoc);
	      double stretch = ((double)((nColor-1)) / (max - min ));
	      Q = I - min;
	      Q = Q * stretch;
	}
	else {
	    QuantizationMSB(I, Q, nColor);
	}

	Size newSize(I.rows+((dX+1)*2), I.cols+((dY+1)*2));
	Mat novaQ(newSize, CV_8U, 1);
	copyMakeBorder(Q, novaQ, (dX+1)*2, (dX+1)*2, (dY+1)*2, (dY+1)*2, BORDER_REPLICATE);
	
	Size imgSize = novaQ.size();
	int height = imgSize.height;
	int width = imgSize.width;
	
	for (i = dX; i < height-dX; i++){
		for (j = dY; j < width-dY; j++){
			int pref = novaQ.at<uchar>(i,j);
			int pviz = novaQ.at<uchar>((i+dX),(j+dY));
			Cm[pref][pviz]++;
		}
	}
	
	double sum = 0;

	for (i = 0; i < nColor; i++){
		for (j = 0; j < nColor; j++){	
			sum += Cm[i][j];
		}
	}

	for (i = 0; i < nColor; i++){
		for (j = 0; j < nColor; j++){
			Cm[i][j] /= sum;
		}
	}
}

void Haralick6(double **Cm, int nColor, Mat &features){
	
	int i,j;
	double m_r = 0.0;
	double m_c = 0.0;
	double s_r = 0.0;
	double s_c = 0.0;
	double *Pi = (double *) calloc(nColor, sizeof(double));
	double *Pj = (double *) calloc(nColor, sizeof(double));
	
	for (i = 0; i < nColor; i++){
		for (j = 0; j<nColor; j++){
			Pi[i] += Cm[i][j];
		}
		m_r += i*Pi[i];	
	}
	
	for (j = 0; j < nColor; j++){
		for (i = 0; i<nColor; i++){
			Pj[j] += Cm[i][j];
		}
		m_c += j*Pj[j];	
	}
	
	for (i = 0; i < nColor; i++){
		s_r += ((i-m_r)*(i-m_r)) * Pi[i];
		s_c += ((i-m_c)*(i-m_c)) * Pj[i];
	}
	s_r = sqrt(s_r);
	s_c = sqrt(s_c);
	
	double maxp = 0.0;
	double corr = 0.0;
	double cont = 0.0;
	double unif = 0.0;
	double homo = 0.0;
	double entr = 0.0;
	double auxv = 0.0;
	
	for (i = 0; i < nColor; i++){
		for (j = 0; j<nColor; j++){
			auxv = Cm[i][j];
			if (maxp < auxv) 
				maxp = auxv;
			if (s_r > 0 && s_c > 0) 
				corr += ((i-m_r)*(j-m_c)*auxv) / (s_r*s_c);	    
			cont += ( (i-j)*(i-j)*auxv );
			unif += (auxv*auxv);
			homo += (auxv) / (1 + abs(i-j));
			if (auxv != 0) 
				entr+= auxv*( log2(auxv));
		}
	}
	
	entr = -entr;
	
	features.at<float>(0, 0) = maxp;
	features.at<float>(0, 1) = corr;
	features.at<float>(0, 2) = cont;
	features.at<float>(0, 3) = unif;
	features.at<float>(0, 4) = homo;
	features.at<float>(0, 5) = entr;  
}

void HARALICK(Mat &I, double **Cm, Mat &features, int nColor, int oNorm) {
	CoocurrenceMatrix(I, Cm, nColor, 2, 0);
	Haralick6(Cm, nColor, features);
}

void ACC(Mat &I, Mat &features, int nColor, int oNorm, int *k, int totalk){
	Mat Q(I.size(), CV_8U, 1);
	
	QuantizationMSB(I, Q, nColor);
	
	int i,j, x, y;
	vector<long int> desc(nColor*totalk);
	
	double descNorm = 0;
	
	Size imgSize = Q.size();
	int height = imgSize.height;
	int width = imgSize.width;
	
	int maxdist = 0;
	for (int d = 0; d < totalk; d++) {
	    if (k[d] > maxdist) maxdist = k[d];
	}

	for (int d = 0; d < totalk; d++) {
	    int cd = k[d]; 
	    
	    for (i = cd; i < (height-cd); i++) {
	 		for (j = cd; j < (width-cd); j++) {
		      	x = (i-cd);
				for (y = (j-cd); y <= (j+cd); y++) {
					if (Q.at<uchar>(i,j) == Q.at<uchar>(x,y)){
						int pos = (int)Q.at<uchar>(i,j);
						desc[pos+(d*totalk)]++;
						descNorm++;
					}
				}
				x = (i+cd);
				for (y = (j-cd); y <= (j+cd); y++) {
					if (Q.at<uchar>(i,j) == Q.at<uchar>(x,y)) {
						int pos = (int)Q.at<uchar>(i,j);
						desc[pos+(d*totalk)]++;
						descNorm++;
					}
				}
				y = (i-cd);
				for (x = (i-cd); x <= (i+cd); x++) {
					if (Q.at<uchar>(i,j) == Q.at<uchar>(x,y)) {
						int pos = (int)Q.at<uchar>(i,j);
						desc[pos+(d*totalk)]++;
						descNorm++;
					}
				}
				y = (i+cd);
				for (x = (i-cd); x <= (i+cd); x++) {
					if (Q.at<uchar>(i,j) == Q.at<uchar>(x,y)) {
						int pos = (int)Q.at<uchar>(i,j);
						desc[pos+(d*totalk)]++;
						descNorm++;
					}
				}
			}
	    }
	}

	vector<float> norm(nColor*totalk);
	float descsum = 0;
	for (i = 0; i < (nColor*totalk) ; i++){
		norm[i] = (float)(desc[i]/(float)descNorm);
		descsum += norm[i];
	}
	if (oNorm == 0) {
		for (i = 0; i < nColor*totalk ; i++) 
			features.at<float>(0,i) = (float)desc[i];
	} else if (oNorm == 1) {
	  for (i = 0; i < nColor*totalk ; i++) 
		features.at<float>(0,i) = norm[i];
	}
	else {
	    for (i = 0; i < nColor*totalk; i++) 
			features.at<float>(0,i) = norm[i]*255;// copia no vetor "features"
	}
}
