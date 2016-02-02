import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.objdetect.CascadeClassifier;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by abr on 11-01-16.
 */
public class OpenCVFaceDetector {

    public static void main(String[] args) {
        System.out.println("\nRunning DetectFaceDemo");

        // Create a face detector from the cascade file in the resources directory.
        CascadeClassifier faceDetector = new CascadeClassifier(getOpenCV_Cascade_Classifier());

        //Read the image to detect faces in
        Mat image = Highgui.imread(getImage(), IMREAD_COLOR);


        // Detect faces in the image.
        MatOfRect faceDetectionsMat = new MatOfRect();
        Size minSize = new Size(20, 20);
        Size maxSize = new Size(1000,1000);
        double scaleFactor = 1.1;
        int minNeighbors = 3;
        int flags = 0;
        faceDetector.detectMultiScale(image, faceDetectionsMat,
                                      scaleFactor, minNeighbors, flags,
                                      minSize, maxSize);

        System.out.println(String.format("Detected %s faces", faceDetectionsMat.size()));

        // Draw a bounding box around each face.
        List<Rect> faceDetections = faceDetectionsMat.toList();
        for (Rect faceDetection : faceDetections) {
            Core.rectangle(image, faceDetection.tl(),
                           faceDetection.br(),
                           Scalar.GREEN,
                           1,
                           LINE_8,
                           0);
        }

        // Save the visualized detection.
        String filename = "faceDetection.png";
        System.out.println(String.format("Writing %s", filename));

        Highgui.imwrite(filename, image);
    }

    private static String getImage() {
        return new Thread.currentThread().getContextClassLoader().getResource("AverageMaleFace.jpg").getPath();
    }

    private static BytePointer getOpenCV_Cascade_Classifier() {
        return new BytePointer(Thread.currentThread().getContextClassLoader().getResource("lbpcascade_frontalface.xml").getPath());

    }
}
