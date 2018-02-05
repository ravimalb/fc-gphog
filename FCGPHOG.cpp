// FC-GPHOG Fused-Color Gabor Pyramidal Histogram of Oriented Gradients.
//
// Related Paper: Sinha, Atreyee, Sugata Banerji, and Chengjun Liu. "New color GPHOG descriptors for object and scene image classification." Machine vision and applications 25, no. 2 (2014): 361-375.
//
// Implemented by Ravimal Bandara - ravimal9@gmail.com / ravimalb@uom.lk

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <ctime>
#include <math.h>
#include <iostream>
#include <opencv2/nonfree/features2d.hpp>
#include <tchar.h>

using namespace cv;
using namespace std;

//RGB to DCS color conversion
void convertToDCS(Mat & input, Mat & output){
	int r, g, b;
	double d1, d2, d3;
	output.create(input.rows, input.cols, CV_8UC3);

	int m=0, n=0;	
	for(m=0; m<input.rows; m++)
	{
		for(n=0; n<input.cols; n++)
		{
			Vec3b pixel = input.at<Vec3b>(m,n);
			r = pixel[2];
			g = pixel[1];
			b = pixel[0];
			d1 = (-0.4258)*r + 0.7918*g -0.4378*b;
			d2 = 0.0440*r - 0.5548*g -0.8308*b;
			d3 = 0.1985*r -0.9019*g + 0.3835*b;

			/*-0.4258	0.7918	-0.4378
			0.0440	0.5548	-0.8308
			0.1985	-0.9019	0.3835*/
			output.at<Vec3b>(m,n).val[2] = d1;
			output.at<Vec3b>(m,n).val[1] = d2;
			output.at<Vec3b>(m,n).val[0] = d3;			
		}
	}
}

//RGB to YIQ Color conversion
void convertToYIQ(Mat & input, Mat & output)
{	
	uchar r, g, b;
	double y, i, q;
	output.create(input.rows, input.cols, CV_8UC3);

	int m=0, n=0;	
	for(m=0; m<input.rows; m++)
	{
		for(n=0; n<input.cols; n++)
		{
			Vec3b pixel = input.at<Vec3b>(m,n);
			r = pixel[2];
			g = pixel[1];
			b = pixel[0];
			y = 0.299*r + 0.587*g + 0.114*b;
			i = 0.596*r - 0.275*g - 0.321*b;
			q = 0.212*r - 0.523*g + 0.311*b;

			output.at<Vec3b>(m,n).val[2] = CV_CAST_8U((int)y);
			output.at<Vec3b>(m,n).val[1] = CV_CAST_8U((int)(i));
			output.at<Vec3b>(m,n).val[0] = CV_CAST_8U((int)(q));			
		}
	}
}

//RGB to ORGB Color conversion
void convertToORGB(Mat & input, Mat & output)
{
	output.create(input.rows,input.cols,CV_32FC3);
	float L=0,C1=0,C2=0;
	float Cyb=0,Crg=0;
	for(int x=0;x<input.cols;x++)
		for(int y=0;y<input.rows;y++)
		{
			Vec3b pixel = input.at<Vec3b>(y,x);
			L = 0.2990*pixel.val[2]+ 0.587*pixel.val[1]+ 0.114*pixel.val[0];			
			C1 = 0.5*(pixel.val[2]+pixel.val[1])-pixel.val[0];
			C2 = 0.866*(pixel.val[2]-pixel.val[1]);
			
			float theta = atan2(C1,C2);
			float theta0 = 0;
			if(theta<CV_PI/3.0)
			{
				theta0 = 3/2.0*theta;
			}else 
			{
				theta0 = CV_PI/2.0 + (3/4.0)+(theta - CV_PI/3.0);
			}			
			Cyb=C1*cos(theta0-theta)-C2*sin(theta0-theta);
			Crg=C1*sin(theta0-theta)+C2*cos(theta0-theta);

			output.at<Vec3f>(y,x)[0]=L;
			output.at<Vec3f>(y,x)[1]=Cyb;
			output.at<Vec3f>(y,x)[2]=Crg;
		}
}

//Compute the histogram of oriented gradient descriptors with given number of bins
void computeHOG(Mat & input, Mat & desc, int bins)
{
	int winWidth=0;
	int winHeight=0;
	//calculate the compatible dimension for win size
	winWidth = input.cols - (input.cols%8+16); /*abs(input.cols%16-8);*/
	winHeight = input.rows - (input.rows%8+16);/*abs(input.rows%16-8);*/

	//printf("\nwinDimension %i X %i %\n",winWidth,winHeight);
	HOGDescriptor hog(Size(winWidth,winHeight),Size(16,16),Size(8,8),Size(8,8),bins,0);
	vector<float> ders;
	vector<Point>locs;
	hog.compute(input,ders,Size(winWidth,winHeight),Size(0,0),locs);	

	int blockX = winWidth/8-1;
	int blockY = winHeight/8-1;

	int gradientBinSize = bins;
 
    // prepare data structure: orientation / gradient strenghts for each cell
	int cells_in_x_dir = blockX*2;
    int cells_in_y_dir = blockY*2;
    int totalnrofcells = cells_in_x_dir * cells_in_y_dir;
    float*** gradientStrengths = new float**[cells_in_y_dir];
    int** cellUpdateCounter   = new int*[cells_in_y_dir];
    for (int y=0; y<cells_in_y_dir; y++)
    {
        gradientStrengths[y] = new float*[cells_in_x_dir];
        cellUpdateCounter[y] = new int[cells_in_x_dir];
        for (int x=0; x<cells_in_x_dir; x++)
        {
            gradientStrengths[y][x] = new float[gradientBinSize];
            cellUpdateCounter[y][x] = 0;
 
            for (int bin=0; bin<gradientBinSize; bin++)
                gradientStrengths[y][x][bin] = 0.0;
        }
    }
 
    // nr of blocks = nr of cells - 1
    // since there is a new block on each cell (overlapping blocks!) but the last one
    int blocks_in_x_dir = cells_in_x_dir - 1;
    int blocks_in_y_dir = cells_in_y_dir - 1;
 
    // compute gradient strengths per cell
    int descriptorDataIdx = 0;
    int cellx = 0;
    int celly = 0;
 
    for (int blockx=0; blockx<blocks_in_x_dir; blockx+=2)
    {
        for (int blocky=0; blocky<blocks_in_y_dir; blocky+=2)            
        {
            // 4 cells per block ...
            for (int cellNr=0; cellNr<4; cellNr++)
            {
                // compute corresponding cell nr
                int cellx = blockx;
                int celly = blocky;
                if (cellNr==1) celly++;
                if (cellNr==2) cellx++;
                if (cellNr==3)
                {
                    cellx++;
                    celly++;
                }
 
                for (int bin=0; bin<gradientBinSize; bin++)
                {
                    float gradientStrength = ders[ descriptorDataIdx ];
                    descriptorDataIdx++;
 
                    gradientStrengths[celly][cellx][bin] += gradientStrength;					
 
                } // for (all bins)
 
 
                //for overlapping blocks lead to multiple updates of this sum!               
                cellUpdateCounter[celly][cellx]++;
            } // for (all cells)
 
        } // for (all block x pos)
    } // for (all block y pos)
 
 
    // compute average gradient strengths
    for (int celly=0; celly<cells_in_y_dir; celly++)
    {
        for (int cellx=0; cellx<cells_in_x_dir; cellx++)
        {
            float NrUpdatesForThisCell = (float)cellUpdateCounter[celly][cellx];						
            // compute average gradient strenghts for each gradient bin direction
            for (int bin=0; bin<gradientBinSize; bin++)
            {
                gradientStrengths[celly][cellx][bin] /= NrUpdatesForThisCell;				
            }
        }
    }

	desc.create(bins,1,CV_32FC1);
	desc = Scalar(0);
	//Level-0 
	for (int celly=0; celly<cells_in_y_dir; celly++)
    {
        for (int cellx=0; cellx<cells_in_x_dir; cellx++)
        {
			for (int bin=0; bin<gradientBinSize; bin++)
            {
				desc.at<float>(bin,0)+= gradientStrengths[celly][cellx][bin];
            }
		}
	}	

	normalize(desc,desc,1,0,NORM_L2,CV_32F);

	// free memory allocated by helper data structures!
    for (int y=0; y<cells_in_y_dir; y++)
    {
      for (int x=0; x<cells_in_x_dir; x++)
      {
           delete[] gradientStrengths[y][x];            
      }
      delete[] gradientStrengths[y];
      delete[] cellUpdateCounter[y];
    }
    delete[] gradientStrengths;
    delete[] cellUpdateCounter;
}

//Compute the Pyramidal Histogram of oriented gradients with given number of pyramid levels.
void computePHOG(Mat & input, Mat & desc, int bins, int levels)
{	
	desc.create(bins,1,CV_32FC1);
	Mat tempDescriptor;
	
	int descIndex=0;
	//Level-0 
	computeHOG(input, tempDescriptor, bins);
	for(int i=0;i<bins;i++)
	{
		desc.at<float>(i,0)=tempDescriptor.at<float>(i,0);		
		descIndex++;
	}
	
	//Build Pyramid levels
	for(int level=0;level<levels;level++){
		int divider = (int)pow(2.0,level+1);	
		int w = input.cols/divider;
		int h = input.rows/divider;
		int rows = (input.rows/h)*h;
		int cols = (input.cols/w)*w;
		for(int y=0;y<rows;y+=h)
			for(int x=0;x<cols;x+=w)
			{
				//tempDescriptor.release();
				computeHOG(input(Rect(x,y,w,h)), tempDescriptor, bins);				
				desc.push_back(tempDescriptor);
			}
	}				
}

//ZScore normalization
void normalizeToZScore(Mat & input)
{
	Scalar_<double> meanA;
	Scalar_<double> stddevA;
	meanStdDev(input,meanA,stddevA);
	double mean = meanA.val[0];
	double stddev = stddevA.val[0];
	for(int y=0;y<input.rows;y++)
		//for(int x=0;x<input.cols;x++)
		{
			input.at<float>(y,0) = (input.at<float>(y,0)- mean)/stddev;
		}
}

//Compute the Gabor Pyramidal Histogram of Oriented Gradient.
void computGPHOG(Mat & input, Mat & desc, int bins, int levels)
{	
	//prepare symentic gabor filters
	Mat gaborKernels[6];
	gaborKernels[0]=getGaborKernel(Size(17,17),8,CV_PI*0,16,1,0);
	gaborKernels[1]=getGaborKernel(Size(17,17),8,CV_PI/6,16,1,0);
	gaborKernels[2]=getGaborKernel(Size(17,17),8,CV_PI/3,16,1,0);
	gaborKernels[3]=getGaborKernel(Size(17,17),8,CV_PI/2,16,1,0);
	gaborKernels[4]=getGaborKernel(Size(17,17),8,2*CV_PI/3,16,1,0);
	gaborKernels[5]=getGaborKernel(Size(17,17),8,5*CV_PI/6,16,1,0);

	Mat gFilteredImages[6];
	Mat tempDesc;
	desc.create(0,1,CV_32FC1);
	input.convertTo(input,CV_32FC1,255,0);

	for(int g=0;g<6;g++)
	{
		filter2D(input, gFilteredImages[g], CV_32F, gaborKernels[g]);
		
		normalize(gFilteredImages[g],gFilteredImages[g],255,0,NORM_L2,CV_32F);		
		gFilteredImages[g].convertTo(gFilteredImages[g],CV_8UC1,255,0);
		computePHOG(gFilteredImages[g], tempDesc, bins, levels);		
		desc.push_back(tempDesc);
		tempDesc.release();
	}
	
	//normalize(desc,desc,0,1,NORM_MINMAX,CV_32F);
	//normalizeToZScore(desc);		
}

//Compute the Fused-Color Gabor Pyramidal Histogram of Oriented Gradient.
void computFCGPHOG(Mat & input, Mat & desc, int bins, int levels)
{
	
	desc.create(0,1,CV_32FC1);
	Mat tempImg = input.clone();
	Mat tempComps[3];
	Mat colorComps[18];
	int index=0;
	//RGB
	split(tempImg,tempComps);
	colorComps[index++]=tempComps[0];
	colorComps[index++]=tempComps[1];
	colorComps[index++]=tempComps[2];
	
	//HSV
	cvtColor(input,tempImg,CV_BGR2HSV);
	split(tempImg,tempComps);
	colorComps[index++]=tempComps[0];
	colorComps[index++]=tempComps[1];
	colorComps[index++]=tempComps[2];

	//oRGB
	convertToORGB(input,tempImg);	
	split(tempImg,tempComps);
	colorComps[index++]=tempComps[0];
	colorComps[index++]=tempComps[1];
	colorComps[index++]=tempComps[2];

	//YCbCr
	cvtColor(input,tempImg,CV_BGR2YCrCb);	
	split(tempImg,tempComps);
	colorComps[index++]=tempComps[0];
	colorComps[index++]=tempComps[1];
	colorComps[index++]=tempComps[2];

	//DCS
	convertToDCS(input,tempImg);	
	split(tempImg,tempComps);
	colorComps[index++]=tempComps[0];
	colorComps[index++]=tempComps[1];
	colorComps[index++]=tempComps[2];

	//YIQ
	convertToYIQ(input,tempImg);	
	split(tempImg,tempComps);
	colorComps[index++]=tempComps[0];
	colorComps[index++]=tempComps[1];
	colorComps[index++]=tempComps[2];

	int colourCompIndex = 0;
	Mat tempDesc[3];
	Mat tempColorDesc;
	Mat tempPCADesc;	

	for(int i=0;i<6;i++)
	{			
		printf(".",i);
		for(int j=0;j<3;j++)
		{
			computGPHOG(colorComps[colourCompIndex++], tempDesc[j], bins, levels);			
			//tempColorDesc.push_back(tempDesc);						
		}					

		tempColorDesc.create(tempDesc[0].rows,3,CV_32FC1);
		for(int j=0;j<3;j++)
		{
			for(int k=0;k<tempDesc[j].rows;k++)
			{
				tempColorDesc.at<float>(k,j)=tempDesc[j].at<float>(k,0);
			}
		}
		//printf("\nDescriptor dimension %d\n",tempColorDesc.cols);
		PCA pca(tempColorDesc,Mat(),CV_PCA_DATA_AS_ROW, 1); 
		pca.project(tempColorDesc,tempPCADesc);
		//printf("\nDescriptor dimension %d x %d\n",tempPCADesc.rows, tempPCADesc.cols);
		desc.push_back(tempPCADesc);	
		tempColorDesc.release();
	}
}

int _tmain(int argc, _TCHAR* argv[])
{	
	FileStorage fs("GPHOGdescriptor.yml", FileStorage::WRITE);
	char * filename = new char[100];		
	char * ymlTag = new char[10];	
	cin>>filename;
	cin>>ymlTag;
	input = imread(filename, CV_LOAD_IMAGE_COLOR); 
	Mat output;
	
	//Compute FCGPHOG descriptor
	computFCGPHOG(input,output, 16, 2);
	printf("\nCompleted! %i X %i\n",output.rows,output.cols);
	
	fs << ymlTag << output;
	fs.release();
	
	//To visualize the descriptor
	Mat draw;
	normalize(output,draw,-1,1,NORM_MINMAX,CV_32FC1);
	Mat visualize(500,20000,CV_8UC1);
	int middle=250;
	for(int i=0;i<15000;i++)
	{
		float val = draw.at<float>(i,0)*250+250;			
		line(visualize,Point(i/4,250),Point(i/4,(int)(val)),Scalar(255,231,100));
	}
	imshow("Graph",visualize);
	waitKey(0);		
	

    return 0;
}

