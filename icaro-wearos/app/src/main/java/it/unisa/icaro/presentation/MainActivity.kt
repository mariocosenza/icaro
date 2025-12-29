package it.unisa.icaro.presentation

import android.Manifest
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.media.RingtoneManager
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.runtime.Composable
import androidx.compose.runtime.mutableStateOf
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.core.splashscreen.SplashScreen.Companion.installSplashScreen
import androidx.wear.compose.material.Button
import androidx.wear.compose.material.MaterialTheme
import androidx.wear.compose.material.Text
import androidx.wear.compose.material.TimeText
import androidx.wear.tooling.preview.devices.WearDevices
import com.google.firebase.messaging.FirebaseMessaging
import it.unisa.icaro.presentation.theme.IcaroTheme

class MainActivity : ComponentActivity() {

    // State variable to trigger UI changes
    private var isFallDetected = mutableStateOf(false)
    private var heartRate = mutableStateOf<Float?>(null)

    // Standard Broadcast Receiver
    private val messageReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            if (intent?.action == "FALL_EVENT") {
                isFallDetected.value = true
                playAlertSound()
            } else if (intent?.action == "HEART_RATE_UPDATE") {
                val hr = intent.getFloatExtra("heart_rate", 0f)
                if (hr > 0) {
                    heartRate.value = hr
                }
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        installSplashScreen()
        super.onCreate(savedInstanceState)
        setTheme(android.R.style.Theme_DeviceDefault)

        // Request Permissions
        if (checkSelfPermission(Manifest.permission.BODY_SENSORS) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(Manifest.permission.BODY_SENSORS), 1)
        }

        setContent {
            WearApp(
                fallDetected = isFallDetected.value,
                heartRate = heartRate.value,
                onReset = {
                    isFallDetected.value = false
                    heartRate.value = null
                }
            )
        }

        // Subscribe to Topic
        FirebaseMessaging.getInstance().subscribeToTopic("fall")
            .addOnCompleteListener { task ->
                if (task.isSuccessful) {
                    Log.d("FIREBASE", "Subscribed to fall topic")
                }
            }
    }

    private fun playAlertSound() {
        try {
            val notification = RingtoneManager.getDefaultUri(RingtoneManager.TYPE_NOTIFICATION)
            val r = RingtoneManager.getRingtone(applicationContext, notification)
            r.play()
        } catch (e: Exception) {
            Log.e("ICARO_MAIN", "Error playing sound", e)
        }
    }

    override fun onResume() {
        super.onResume()
        val filter = IntentFilter().apply {
            addAction("FALL_EVENT")
            addAction("HEART_RATE_UPDATE")
        }
        registerReceiver(messageReceiver, filter, Context.RECEIVER_NOT_EXPORTED)
    }

    override fun onPause() {
        super.onPause()
        unregisterReceiver(messageReceiver)
    }
}

@Composable
fun WearApp(fallDetected: Boolean, heartRate: Float?, onReset: () -> Unit) {
    IcaroTheme {
        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(MaterialTheme.colors.background),
            contentAlignment = Alignment.Center
        ) {
            TimeText()
            Column(
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.Center,
                modifier = Modifier.fillMaxSize()
            ) {
                Greeting(fallDetected)
                if (fallDetected) {
                    if (heartRate != null) {
                        Spacer(modifier = Modifier.height(8.dp))
                        Text(
                            text = "HR: ${heartRate.toInt()} bpm",
                            color = Color.White,
                            style = MaterialTheme.typography.body1
                        )
                    }
                    Spacer(modifier = Modifier.height(16.dp))
                    Button(onClick = onReset) {
                        Text("I'm OK")
                    }
                }
            }
        }
    }
}

@Composable
fun Greeting(fallDetected: Boolean) {
    val textToShow = if (fallDetected) "FALL DETECTED!" else "Monitoring..."
    val textColor = if (fallDetected) Color.Red else MaterialTheme.colors.primary

    Text(
        modifier = Modifier.fillMaxWidth(),
        textAlign = TextAlign.Center,
        color = textColor,
        text = textToShow,
        style = MaterialTheme.typography.title2
    )
}

@Preview(device = WearDevices.SMALL_ROUND, showSystemUi = true)
@Composable
fun DefaultPreview() {
    WearApp(fallDetected = true, heartRate = 85f, onReset = {})
}
