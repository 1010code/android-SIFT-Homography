package com.example.androidsifthomography;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.media.MediaScannerConnection;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;

import org.opencv.android.Utils;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.DMatch;
import org.opencv.core.KeyPoint;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.Features2d;
import org.opencv.features2d.SIFT;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;


public class MainActivity extends AppCompatActivity {

    // Used to load the 'native-lib' library on application startup.
    static {
        System.loadLibrary("opencv_java4");
    }
    private ImageView imageView;
    private Bitmap inputImage, targetImage; // make bitmap from image resource
    private  TextView tv;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Show result Good Match Found
        tv = findViewById(R.id.sample_text);

        inputImage = BitmapFactory.decodeResource(getResources(), R.drawable.coca_cola);
        targetImage = BitmapFactory.decodeResource(getResources(), R.drawable.coca_cola_test8);
        imageView = (ImageView) this.findViewById(R.id.imageView);
        sift();

    }
    public void sift() {
        /***
         * Loading Image
         * source & target Image need converting color to grayscale */
        // source
        Mat imgObject = new Mat();
        Utils.bitmapToMat(inputImage, imgObject);
        Imgproc.cvtColor(imgObject, imgObject, Imgproc.COLOR_RGBA2GRAY);
        // target
        Mat imgScene = new Mat();
        Utils.bitmapToMat(targetImage, imgScene);
        // For output RGBA with target Image
        Imgproc.cvtColor(imgScene, imgScene, Imgproc.COLOR_RGBA2GRAY);

        /*** Step 1: Detect the keypoints using SIFT Detector, compute the descriptors */
        SIFT sift = SIFT.create();
        MatOfKeyPoint keypointsObject = new MatOfKeyPoint(), keypointsScene = new MatOfKeyPoint();
        Mat descriptorsObject = new Mat(), descriptorsScene = new Mat();
        sift.detectAndCompute(imgObject, new Mat(), keypointsObject, descriptorsObject);
        sift.detectAndCompute(imgScene, new Mat(), keypointsScene, descriptorsScene);
        /***  Step 2: Matching descriptor vectors with a FLANN based matcher */
        DescriptorMatcher matcher = DescriptorMatcher.create(DescriptorMatcher.FLANNBASED);
        List<MatOfDMatch> knnMatches = new ArrayList<>();
        matcher.knnMatch(descriptorsObject, descriptorsScene, knnMatches, 2);
        //-- Filter matches using the Lowe's ratio test
        float ratioThresh = 0.75f;
        List<DMatch> listOfGoodMatches = new ArrayList<>();
        for (int i = 0; i < knnMatches.size(); i++) {
            if (knnMatches.get(i).rows() > 1) {
                DMatch[] matches = knnMatches.get(i).toArray();
                if (matches[0].distance < ratioThresh * matches[1].distance) {
                    listOfGoodMatches.add(matches[0]);
                }
            }
        }
        MatOfDMatch goodMatches = new MatOfDMatch();
        goodMatches.fromList(listOfGoodMatches);
        Log.e("GoodMatch",listOfGoodMatches.size() + " GoodMatch Found");
        // Show result Good Match Found
        tv.setText(listOfGoodMatches.size()+" Good Match Found");
        /*** Draw matches */
        Utils.bitmapToMat(inputImage, imgObject);
        Utils.bitmapToMat(targetImage, imgScene);
        Mat imgMatches = new Mat();
        Features2d.drawMatches(imgObject, keypointsObject, imgScene, keypointsScene, goodMatches, imgMatches, Scalar.all(-1),
                Scalar.all(-1), new MatOfByte(), Features2d.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS);
        Imgproc.cvtColor(imgMatches, imgMatches, Imgproc.COLOR_RGBA2BGR);
        //-- Localize the object
        List<Point> obj = new ArrayList<>();
        List<Point> scene = new ArrayList<>();
        List<KeyPoint> listOfKeypointsObject = keypointsObject.toList();
        List<KeyPoint> listOfKeypointsScene = keypointsScene.toList();
        for (int i = 0; i < listOfGoodMatches.size(); i++) {
            //-- Get the keypoints from the good matches
            obj.add(listOfKeypointsObject.get(listOfGoodMatches.get(i).queryIdx).pt);
            scene.add(listOfKeypointsScene.get(listOfGoodMatches.get(i).trainIdx).pt);
        }
        MatOfPoint2f objMat = new MatOfPoint2f(), sceneMat = new MatOfPoint2f();
        objMat.fromList(obj);
        sceneMat.fromList(scene);
        double ransacReprojThreshold = 3.0;
        Mat H = Calib3d.findHomography( objMat, sceneMat, Calib3d.RANSAC, ransacReprojThreshold );
        //-- Get the corners from the image_1 ( the object to be "detected" )
        Mat objCorners = new Mat(4, 1, CvType.CV_32FC2), sceneCorners = new Mat();
        float[] objCornersData = new float[(int) (objCorners.total() * objCorners.channels())];
        objCorners.get(0, 0, objCornersData);
        objCornersData[0] = 0;
        objCornersData[1] = 0;
        objCornersData[2] = imgObject.cols();
        objCornersData[3] = 0;
        objCornersData[4] = imgObject.cols();
        objCornersData[5] = imgObject.rows();
        objCornersData[6] = 0;
        objCornersData[7] = imgObject.rows();
        objCorners.put(0, 0, objCornersData);
        Core.perspectiveTransform(objCorners, sceneCorners, H);
        float[] sceneCornersData = new float[(int) (sceneCorners.total() * sceneCorners.channels())];
        sceneCorners.get(0, 0, sceneCornersData);
        //-- Draw lines between the corners (the mapped object in the scene - image_2 )
        Imgproc.line(imgMatches, new Point(sceneCornersData[0] + imgObject.cols(), sceneCornersData[1]),
                new Point(sceneCornersData[2] + imgObject.cols(), sceneCornersData[3]), new Scalar(0, 255, 0), 4);
        Imgproc.line(imgMatches, new Point(sceneCornersData[2] + imgObject.cols(), sceneCornersData[3]),
                new Point(sceneCornersData[4] + imgObject.cols(), sceneCornersData[5]), new Scalar(0, 255, 0), 4);
        Imgproc.line(imgMatches, new Point(sceneCornersData[4] + imgObject.cols(), sceneCornersData[5]),
                new Point(sceneCornersData[6] + imgObject.cols(), sceneCornersData[7]), new Scalar(0, 255, 0), 4);
        Imgproc.line(imgMatches, new Point(sceneCornersData[6] + imgObject.cols(), sceneCornersData[7]),
                new Point(sceneCornersData[0] + imgObject.cols(), sceneCornersData[1]), new Scalar(0, 255, 0), 4);

        Bitmap bitmap=Bitmap.createBitmap(imgMatches.cols(), imgMatches.rows(), Bitmap.Config.ARGB_4444);
        Imgproc.cvtColor(imgMatches, imgMatches, Imgproc.COLOR_RGBA2BGR);
        Utils.matToBitmap(imgMatches, bitmap);
        imageView.setImageBitmap(bitmap);
    }
}