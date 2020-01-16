package com.rishi.digitdetect;

import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.ColorMatrix;
import android.graphics.ColorMatrixColorFilter;
import android.graphics.Paint;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Arrays;

public class MainActivity extends AppCompatActivity {

    Interpreter interpreter;
    Interpreter.Options options;
    ByteBuffer imgData;

    private static final ColorMatrix INVERT = new ColorMatrix(
            new float[]{
                    -1, 0, 0, 0, 255,
                    0, -1, 0, 0, 255,
                    0, 0, -1, 0, 255,
                    0, 0, 0, 1, 0
            });

    private static final ColorMatrix BLACKWHITE = new ColorMatrix(
            new float[]{
                    0.5f, 0.5f, 0.5f, 0, 0,
                    0.5f, 0.5f, 0.5f, 0, 0,
                    0.5f, 0.5f, 0.5f, 0, 0,
                    0, 0, 0, 1, 0,
                    -1, -1, -1, 0, 1
            }
    );

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        AssetFileDescriptor fileDescriptor = null;
        try {
            fileDescriptor = getAssets().openFd("mnist.tflite");
            FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
            FileChannel fileChannel = inputStream.getChannel();
            long startOffset = fileDescriptor.getStartOffset();
            long declaredLength = fileDescriptor.getDeclaredLength();
            MappedByteBuffer byteBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
            interpreter = new Interpreter(byteBuffer);
        } catch (IOException e) {
            System.out.println("EXCEPTION");
            e.printStackTrace();
        }
        CanvasView canvas = findViewById(R.id.canvasView);
        Button detect = findViewById(R.id.detect);
        detect.setOnClickListener(v -> {
             imgData = ByteBuffer.allocateDirect(4 * 28 * 28);
             imgData.order(ByteOrder.nativeOrder());
             Bitmap scaled = prepareImageForClassification(canvas.canvasBitmap);
             convertBitmapToByteBuffer(scaled);
             float[][] result = new float[1][10];
             System.out.println("received " + Arrays.toString(result[0]));
             interpreter.run(imgData, result);
             System.out.println("received " + Arrays.toString(result[0]));
             int maxAt = 0;
             for (int i = 0; i < result[0].length; i++) {
                 maxAt = result[0][i] > result[0][maxAt] ? i : maxAt;
             }
             ((TextView)findViewById(R.id.detection)).setText("Detected a " + maxAt);
        });
        Button clear = findViewById(R.id.clear);
        clear.setOnClickListener(v -> {
            canvas.clear();
        });
    }

    private void convertBitmapToByteBuffer(Bitmap bitmap) {
        if (imgData == null) {
            System.out.println("returning");
            return;
        }
        imgData.rewind();
        int[] pixels = new int[28 * 28];
        bitmap.getPixels(pixels, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        int i = 0;
        System.out.print("[");
        for (int pixel : pixels) {
            float rChannel = (pixel >> 16) & 0xFF;
            float gChannel = (pixel >> 8) & 0xFF;
            float bChannel = (pixel) & 0xFF;
            float pixelValue = (rChannel + gChannel + bChannel) / 3 / 255.f;
            imgData.putFloat(pixelValue);
            System.out.print(pixelValue);
            i++;
            if (i % 28 == 0) {
                System.out.print("], [");
            } else {
                System.out.print(", ");
            }
        }
        System.out.println("num pixels is " + i);
    }

    public static Bitmap prepareImageForClassification(Bitmap bitmap) {
        ColorMatrix colorMatrix = new ColorMatrix();
        colorMatrix.setSaturation(0);
        colorMatrix.postConcat(BLACKWHITE);
        colorMatrix.postConcat(INVERT);
        ColorMatrixColorFilter f = new ColorMatrixColorFilter(colorMatrix);

        Paint paint = new Paint();
        paint.setColorFilter(f);

        System.out.println("dims are " + bitmap.getWidth() + ", " + bitmap.getHeight());

        Bitmap bmpGrayscale = Bitmap.createScaledBitmap(
                bitmap,
               28,
                28,
                false);
        Canvas canvas = new Canvas(bmpGrayscale);
        canvas.drawBitmap(bmpGrayscale, 0, 0, paint);
        return bmpGrayscale;
    }

}
