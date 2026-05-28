package elazarkin.ksg.external.trackertester;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;

import elazarkin.ksg.external.trackertester.activities.AutoCorrelationActivity;
import elazarkin.ksg.external.trackertester.activities.FrameStreamActivity;
import elazarkin.ksg.external.trackertester.activities.ManualDisplacementActivity;
import elazarkin.ksg.external.trackertester.activities.RecursiveTrackingActivity;
import elazarkin.ksg.external.trackertester.databinding.ActivityMainBinding;

public class MainActivity extends AppCompatActivity {

    // Used to load the 'trackertester' library on application startup.
    static {
        System.loadLibrary("trackertester");
    }

    private ActivityMainBinding binding;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        // Set click listeners for dashboard cards
        binding.cardAutoCorrelation.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(MainActivity.this, AutoCorrelationActivity.class);
                startActivity(intent);
            }
        });

        binding.cardManualDisplacement.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(MainActivity.this, ManualDisplacementActivity.class);
                startActivity(intent);
            }
        });

        binding.cardRecursiveTracking.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(MainActivity.this, RecursiveTrackingActivity.class);
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
     * A native method that is implemented by the 'trackertester' native library,
     * which is packaged with this application.
     */
    public native String stringFromJNI();

    public static native void cropAndMaskFrame(
            android.graphics.Bitmap srcBitmap,
            float targetX,
            float targetY,
            float cropScale,
            float maskRadius,
            float[] outBuffer
    );

    public static native float[] calculateCenterOfMass(
            float[] heatmap,
            float threshold
    );
}