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

    public native String stringFromJNI();
}