import org.opencv.contrib.FaceRecognizer;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfInt4;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.List;

import static org.opencv.contrib.Contrib.applyColorMap;
import static org.opencv.core.Core.NORM_MINMAX;
import static org.opencv.core.Core.normalize;
import static org.opencv.highgui.Highgui.imwrite;


public class FisherFaces {

    private static Mat norm_0_255(Mat src) {
        // Create and return normalized image:
        Mat dst = new Mat();
        switch (src.channels()) {
            case 1:
                normalize (src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
                break;
            case 3:
                normalize (src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
                break;
            default:
                src.copyTo(dst);
                break;
        }
        return dst;
    }

    private static void read_csv(String filename, Mat images, MatOfInt labels, String separator) throws IOException {

        BufferedReader file = null;
        try {
            file = new BufferedReader(new InputStreamReader(new FileInputStream(filename)));
        } catch (FileNotFoundException e) {
            String error_message = "No valid input file was given, please check the given filename.";
            throw new IOException(error_message,e);
        }
        String line, path, classlabel;

        int i = 0;
        while ((line = file.readLine()) != null) {

            String[] splits = line.split(separator);
            path = splits[0];
            classlabel = splits[1];
            if (!path.isEmpty() && !classlabel.isEmpty()) {
                images.put(i,imread(path, 0));
                labels.put(i,0,Integer.parseInt(classlabel));
                i++;
            }
        }
    }

    public void main(String[] args){
    
        // Check for valid command line arguments, print usage
        // if no arguments were given.
        if (args.length < 2) {
            System.out.println("usage: " + args[0] + " <csv.ext> <output_folder> ");
            System.exit(1);
        }
        String output_folder = ".";
        if (args.length == 3) {
            output_folder = args[2];
        }
        // Get the path to your CSV.
        String fn_csv = args[1];
        // These vectors hold the images and corresponding labels.
        List<Mat> images;
        MatOfInt4 labels = new MatOfInt4();
        // Read in the data. This can fail if no valid
        // input filename is given.
        try {
            read_csv(fn_csv, images, labels,";");
        } catch (IOException  e){
            System.err.println("Error opening file \"" + fn_csv + "\". Reason: " + e.getMessage());
            // nothing more we can do
            System.exit(1);
        }
        // Quit if there are not enough images for this demo.
        if (images.size() <= 1) {
            String error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
            throw new RuntimeException(error_message);
        }
        // Get the height from the first image. We'll need this
        // later in code to reshape the images to their original
        // size:
        Mat image0 = images.get(0);
        int height = image0.rows();
        // The following lines simply get the last images from
        // your dataset and remove it from the vector. This is
        // done, so that the training data (which we learn the
        // cv::FaceRecognizer on) and the test data we test
        // the model with, do not overlap.
        Mat testSample = images.get(images.size() -1);
        int testLabel =  labels.row(labels.size()-1).;
        labels.
        labels.pop_back();

        // The following lines create an Fisherfaces model for
        // face recognition and train it with the images and
        // labels read from the given CSV file.
        // If you just want to keep 10 Fisherfaces, then call
        // the factory method like this:
        //
        //      cv::createFisherFaceRecognizer(10);
        //
        // However it is not useful to discard Fisherfaces! Please
        // always try to use _all_ available Fisherfaces for
        // classification.
        //
        // If you want to create a FaceRecognizer with a
        // confidence threshold (e.g. 123.0) and use _all_
        // Fisherfaces, then call it with:
        //
        //      cv::createFisherFaceRecognizer(0, 123.0);
        //
        FaceRecognizer model = createFisherFaceRecognizer();
        model.train(images, labels);

        // The following line predicts the label of a given
        // test image:
        int predictedLabel = model.predict(testSample);
        //
        // To get the confidence of a prediction call the model with:
        //
        //      int predictedLabel = -1;
        //      double confidence = 0.0;
        //      model->predict(testSample, predictedLabel, confidence);
        //
        System.out.println(String.format("Predicted class = %d / Actual class = %d.", predictedLabel, testLabel));

        // Here is how to get the eigenvalues of this Eigenfaces model:
        Mat eigenvalues = model.getMat("eigenvalues");
        // And we can do the same to display the Eigenvectors (read Eigenfaces):
        Mat W = model.getMat("eigenvectors");
        // Get the sample mean from the training data
        Mat mean = model.getMat("mean");
        // Display or save:
        if (args.length == 2) {
            imshow("mean", norm_0_255(mean.reshape(1, height)));
        } else {
            imwrite(String.format("%s/mean.png", output_folder), norm_0_255(mean.reshape(1, height)));
        }
        // Display or save the first, at most 16 Fisherfaces:
        for (int i = 0; i < Math.min(16, W.cols()); i++) {
            String msg = String.format("Eigenvalue #%d = %.5f", i, eigenvalues.at <double>(i));
            System.out.println(msg);
            // get eigenvector #i
            Mat ev = W.col(i).clone();
            // Reshape to original size & normalize to [0...255] for imshow.
            Mat grayscale = norm_0_255(ev.reshape(1, height));
            // Show the image & apply a Bone colormap for better sensing.
            Mat cgrayscale;
            applyColorMap(grayscale, cgrayscale, COLORMAP_BONE);
            // Display or save:
            if (args.length == 2) {
                imshow(String.format("fisherface_%d", i), cgrayscale);
            } else {
                imwrite(String.format("%s/fisherface_%d.png", output_folder, i), norm_0_255(cgrayscale));
            }
        }
        // Display or save the image reconstruction at some predefined steps:
        for (int num_component = 0; num_component < Math.min(16, W.cols()); num_component++) {
            // Slice the Fisherface from the model:
            Mat ev = W.col(num_component);
            Mat projection = subspaceProject(ev, mean, image0.reshape(1, 1));
            Mat reconstruction = subspaceReconstruct(ev, mean, projection);
            // Normalize the result:
            reconstruction = norm_0_255(reconstruction.reshape(1, height));
            // Display or save:
            if (args.length == 2) {
                imshow(String.format("fisherface_reconstruction_%d", num_component), reconstruction);
            } else {
                imwrite(String.format("%s/fisherface_reconstruction_%d.png", output_folder, num_component),
                        reconstruction);
            }
        }
        // Display if we are not writing to an output folder:
        if (args.length == 2) {
            waitKey(0);
        }
        System.exit(0);
    }


}