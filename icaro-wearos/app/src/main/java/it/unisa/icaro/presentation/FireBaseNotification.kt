package it.unisa.icaro.presentation

import android.content.Context
import android.content.Intent
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.util.Log
import com.google.firebase.messaging.FirebaseMessagingService
import com.google.firebase.messaging.RemoteMessage

class FirebaseNotificationService : FirebaseMessagingService() {

    override fun onMessageReceived(message: RemoteMessage) {
        super.onMessageReceived(message)

        Log.d("FIREBASE_DEBUG", "Message received from: ${message.from}")
        Log.d("FIREBASE_DEBUG", "Data payload: ${message.data}")

        val intent = Intent("FALL_EVENT")
        intent.setPackage(packageName)
        sendBroadcast(intent)

        measureAndSendHeartRate()
    }

    private fun measureAndSendHeartRate() {
        val sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        val heartRateSensor = sensorManager.getDefaultSensor(Sensor.TYPE_HEART_RATE)

        if (heartRateSensor == null) {
            Log.e("ICARO_SERVICE", "No Heart Rate Sensor found")
            return
        }

        val readings = mutableListOf<Float>()

        val listener = object : SensorEventListener {
            override fun onSensorChanged(event: SensorEvent?) {
                if (event != null && event.values.isNotEmpty()) {
                    readings.add(event.values[0])
                }
            }
            override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
        }

        sensorManager.registerListener(listener, heartRateSensor, SensorManager.SENSOR_DELAY_NORMAL)

        try {
            Thread.sleep(2000)
        } catch (e: InterruptedException) {
            e.printStackTrace()
        }

        sensorManager.unregisterListener(listener)

        if (readings.isNotEmpty()) {
            val meanValue = readings.average()
            sendHeartRateToApi(meanValue)
        } else {
            Log.w("ICARO_SERVICE", "No data collected")
        }
    }

    private fun sendHeartRateToApi(meanHeartbeat: Double) {
        try {
            val targetUrl = "http://192.168.1.15:8000/api/v1/measure/${meanHeartbeat.toInt()}"
            val response = khttp.post(url = targetUrl)

            if (response.statusCode == 200) {
                Log.d("ICARO_SERVICE", "Success: ${response.text}")
            } else {
                Log.e("ICARO_SERVICE", "Failed: ${response.statusCode}")
            }
        } catch (e: Exception) {
            Log.e("ICARO_SERVICE", "Network Error", e)
        }
    }

    override fun onNewToken(token: String) {}
}