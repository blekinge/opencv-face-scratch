import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.RectVector;
import org.bytedeco.javacpp.opencv_core.Scalar;
import org.bytedeco.javacpp.opencv_objdetect.CascadeClassifier;

import static org.bytedeco.javacpp.opencv_imgcodecs.IMREAD_COLOR;
import static org.bytedeco.javacpp.opencv_imgcodecs.imread;
import static org.bytedeco.javacpp.opencv_imgcodecs.imwrite;
import static org.bytedeco.javacpp.opencv_imgproc.rectangle;

/**
 * Created by abr on 11-01-16.
 */
public class OpenCVFaceDetector {

    public static void main(String[] args) {
        System.out.println("\nRunning DetectFaceDemo");

        // Create a face detector from the cascade file in the resources directory.
        CascadeClassifier faceDetector = new CascadeClassifier(getOpenCV_Cascade_Classifier());

        //Read the image to detect faces in
        Mat image = imread(getImage(),IMREAD_COLOR);


        // Detect faces in the image.
        RectVector faceDetections = new RectVector();
        opencv_core.Size minSize = new opencv_core.Size(20, 20);
        opencv_core.Size maxSize = new opencv_core.Size(1000,1000);
        double scaleFactor = 1.1;
        int minNeighbors = 3;
        int flags = 0;
        faceDetector.detectMultiScale(image, faceDetections,
                                      scaleFactor, minNeighbors, flags,
                                      minSize, maxSize);

        System.out.println(String.format("Detected %s faces", faceDetections.size()));

        // Draw a bounding box around each face.
        for (int i = 0; i < faceDetections.size(); i++) {
            opencv_core.Rect rect = faceDetections.get(i);
            rectangle(image, rect.tl(),
                      rect.br(),
                      Scalar.GREEN,
                      1,
                      opencv_core.LINE_8,
                      0);
        }

        // Save the visualized detection.
        String filename = "faceDetection.png";
        System.out.println(String.format("Writing %s", filename));
        IntPointer params = new IntPointer();
        imwrite(new BytePointer(filename), image, params);
    }

    private static BytePointer getImage() {
        return new BytePointer(Thread.currentThread().getContextClassLoader().getResource("AverageMaleFace.jpg").getPath());
    }

    private static BytePointer getOpenCV_Cascade_Classifier() {
        return new BytePointer(Thread.currentThread().getContextClassLoader().getResource("lbpcascade_frontalface.xml").getPath());

    }
}
