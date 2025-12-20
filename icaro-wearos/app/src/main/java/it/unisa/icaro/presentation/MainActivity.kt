package it.unisa.icaro.presentation

import android.Manifest
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.runtime.Composable
import androidx.compose.runtime.mutableStateOf
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.tooling.preview.Preview
import androidx.core.splashscreen.SplashScreen.Companion.installSplashScreen
import androidx.wear.compose.material.MaterialTheme
import androidx.wear.compose.material.Text
import androidx.wear.compose.material.TimeText
import androidx.wear.tooling.preview.devices.WearDevices
import com.google.firebase.messaging.FirebaseMessaging
import it.unisa.icaro.presentation.theme.IcaroTheme

class MainActivity : ComponentActivity() {

    // State variable to trigger UI changes
    private var isFallDetected = mutableStateOf(false)

    // Standard Broadcast Receiver
    private val messageReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            isFallDetected.value = true
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
            WearApp(isFallDetected.value)
        }

        // Subscribe to Topic
        FirebaseMessaging.getInstance().subscribeToTopic("fall")
            .addOnCompleteListener { task ->
                if (task.isSuccessful) {
                    Log.d("FIREBASE", "Subscribed to fall topic")
                }
            }
    }

    override fun onResume() {
        super.onResume()
        // Register receiver compatible with newer Android versions
        val filter = IntentFilter("FALL_EVENT")
        if (Build.VERSION.SDK_INT >= 33) {
            registerReceiver(messageReceiver, filter, Context.RECEIVER_NOT_EXPORTED)
        } else {
            registerReceiver(messageReceiver, filter)
        }
    }

    override fun onPause() {
        super.onPause()
        unregisterReceiver(messageReceiver)
    }
}

@Composable
fun WearApp(fallDetected: Boolean) {
    IcaroTheme {
        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(MaterialTheme.colors.background),
            contentAlignment = Alignment.Center
        ) {
            TimeText()
            Greeting(fallDetected)
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
        text = textToShow
    )
}

@Preview(device = WearDevices.SMALL_ROUND, showSystemUi = true)
@Composable
fun DefaultPreview() {
    WearApp(fallDetected = true)
}