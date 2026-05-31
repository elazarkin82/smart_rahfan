package elazarkin.ksg.tracker3_lite;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;

import elazarkin.ksg.tracker3_lite.activities.AutoCorrelationActivity;
import elazarkin.ksg.tracker3_lite.activities.FrameStreamActivity;
import elazarkin.ksg.tracker3_lite.databinding.ActivityMainBinding;

public class MainActivity extends AppCompatActivity {

    // Used to load the 'tracker3_lite' library on application startup.
    static {
        System.loadLibrary("tracker3_lite");
    }

    private ActivityMainBinding binding;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        // Set click listeners for the two dashboard cards
        binding.cardAutoCorrelation.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(MainActivity.this, AutoCorrelationActivity.class);
                startActivity(intent);
            }
        });

        binding.cardFrameStream.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(MainActivity.this, FrameStreamActivity.class);
                startActivity(intent);
            }
        });
    }

    /**
     * Native JNI pre-processing function that downsamples the Bitmap to 256x256 and applies an optional attention mask.
     */
    public static native void downsampleAndMaskFrameV3(
            android.graphics.Bitmap srcBitmap,
            float targetX,
            float targetY,
            float maskRadius,
            boolean useExponentialMask,
            float maskSigma,
            boolean isSearchFrame,
            int numChannels,
            float[] outBuffer
    );

    /**
     * Native JNI post-processing function that calculates the Center of Mass (CoM) on the predicted 64x64 heatmap.
     */
    public static native float[] calculateCenterOfMass(
            float[] heatmap,
            float threshold
    );

    /**
     * Maps screen touch coordinates (viewX, viewY) relative to the PreviewView
     * into normalized [0, 1] coordinates inside the captured frame Bitmap using FIT_CENTER.
     */
    public static float[] mapScreenCoordsToFrame(float viewX, float viewY, int viewWidth, int viewHeight, int imgWidth, int imgHeight) {
        if (viewWidth <= 0 || viewHeight <= 0 || imgWidth <= 0 || imgHeight <= 0) return null;
        
        float viewRatio = (float) viewWidth / viewHeight;
        float imgRatio = (float) imgWidth / imgHeight;
        
        float scaleX = 1.0f;
        float scaleY = 1.0f;
        float offsetX = 0.0f;
        float offsetY = 0.0f;
        
        if (imgRatio > viewRatio) { // Fit Width, height is letterboxed
            float actualHeight = viewWidth / imgRatio;
            offsetY = (viewHeight - actualHeight) / 2.0f;
            scaleX = (float) imgWidth / viewWidth;
            scaleY = (float) imgHeight / actualHeight;
        } else { // Fit Height, width is letterboxed
            float actualWidth = viewHeight * imgRatio;
            offsetX = (viewWidth - actualWidth) / 2.0f;
            scaleX = (float) imgWidth / actualWidth;
            scaleY = (float) imgHeight / viewHeight;
        }
        
        float bmpX = (viewX - offsetX) * scaleX;
        float bmpY = (viewY - offsetY) * scaleY;
        
        if (bmpX >= 0 && bmpX < imgWidth && bmpY >= 0 && bmpY < imgHeight) {
            return new float[]{ bmpX / imgWidth, bmpY / imgHeight };
        }
        return null;
    }

    public native String stringFromJNI();
}