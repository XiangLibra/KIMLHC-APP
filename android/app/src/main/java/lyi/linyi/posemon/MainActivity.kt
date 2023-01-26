/* Copyright 2022 Lin Yi. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================
*/

/** 本應用主要對Tensorflow Lite Pose Estimation 示例项目的 MainActivity.kt
 *文件進行了重寫，示例項目中其餘文件去掉了包名調整外基本無改動，原版權歸
 *  The Tensorflow Authors 所有 */

package lyi.linyi.posemon

import android.Manifest
import android.app.AlertDialog
import android.app.Dialog
import android.content.pm.PackageManager
import android.media.MediaPlayer
import android.os.Bundle
import android.os.Process
import android.view.SurfaceView
import android.view.View
import android.view.WindowManager
import android.widget.*
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.fragment.app.DialogFragment
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import lyi.linyi.posemon.camera.CameraSource
import lyi.linyi.posemon.data.Device
import lyi.linyi.posemon.data.Camera
import lyi.linyi.posemon.ml.ModelType
import lyi.linyi.posemon.ml.MoveNet
import lyi.linyi.posemon.ml.PoseClassifier

class MainActivity : AppCompatActivity() {
    companion object {
        private const val FRAGMENT_DIALOG = "dialog"
    }

    /**為視頻畫面創作一個 SurfaceView */
    private lateinit var surfaceView: SurfaceView

    /** 修改默認計算設備：CPU、GPU、NNAPI（AI加速器） */
    private var device = Device.CPU
    /**修改默認頭像：FRONT、BACK */
    private var selectedCamera = Camera.BACK

    /**定義幾個計數器*/

    private var standardCounter = 0
    private var missingCounter = 0
    private var KIM1Counter = 0
    private var KIM2Counter = 0
    private var KIM3Counter = 0
    private var KIM4Counter = 0

    /** 定義一個歷史姿態寄存器 */
    private var poseRegister = "standard"

    /**設置一個用來顯示Debug 信息的 TextView */
    private lateinit var tvDebug: TextView

    /**設置一個用來顯示當前坐姿狀態的ImageView */
    private lateinit var ivStatus: ImageView

    private lateinit var tvFPS: TextView
    private lateinit var tvScore: TextView
    private lateinit var spnDevice: Spinner
    private lateinit var spnCamera: Spinner

    private var cameraSource: CameraSource? = null
    private var isClassifyPose = true

    private val requestPermissionLauncher =
        registerForActivityResult(
            ActivityResultContracts.RequestPermission()
        ) { isGranted: Boolean ->
            if (isGranted) {
                /**得到用戶相機授權後，程序開始運行 */
                openCamera()
            } else {
                /** 提示用戶“未獲得相機權限制，應用無法運行” */
                ErrorDialog.newInstance(getString(R.string.tfe_pe_request_permission))
                    .show(supportFragmentManager, FRAGMENT_DIALOG)
            }
        }

    private var changeDeviceListener = object : AdapterView.OnItemSelectedListener {
        override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
            changeDevice(position)
        }

        override fun onNothingSelected(parent: AdapterView<*>?) {
            /**如果使用用戶未選擇運算設備，使用默認設備進行計算 */
        }
    }

    private var changeCameraListener = object : AdapterView.OnItemSelectedListener {
        override fun onItemSelected(p0: AdapterView<*>?, view: View?, direction: Int, id: Long) {
            changeCamera(direction)
        }

        override fun onNothingSelected(p0: AdapterView<*>?) {
            /** 如果使用用戶未選擇攝像頭，使用默認攝像頭進行拍攝*/
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        /**程序運行時保持畫面常亮*/
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        tvScore = findViewById(R.id.tvScore)

        /**用來顯示Debug 信息 */
        tvDebug = findViewById(R.id.tvDebug)

        /**用來顯示當前坐姿狀態 */
        ivStatus = findViewById(R.id.ivStatus)

        tvFPS = findViewById(R.id.tvFps)
        spnDevice = findViewById(R.id.spnDevice)
        spnCamera = findViewById(R.id.spnCamera)
        surfaceView = findViewById(R.id.surfaceView)
        initSpinner()
        if (!isCameraPermissionGranted()) {
            requestPermission()
        }
    }

    override fun onStart() {
        super.onStart()
        openCamera()
    }

    override fun onResume() {
        cameraSource?.resume()
        super.onResume()
    }

    override fun onPause() {
        cameraSource?.close()
        cameraSource = null
        super.onPause()
    }

    /**檢查相機權限是否有權限*/
    private fun isCameraPermissionGranted(): Boolean {
        return checkPermission(
            Manifest.permission.CAMERA,
            Process.myPid(),
            Process.myUid()
        ) == PackageManager.PERMISSION_GRANTED
    }

    private fun openCamera() {
        /** 音頻播放 */



        if (isCameraPermissionGranted()) {
            if (cameraSource == null) {
                cameraSource =
                    CameraSource(surfaceView, selectedCamera, object : CameraSource.CameraSourceListener {
                        override fun onFPSListener(fps: Int) {

                            /**解釋一下，tfe_pe_tv 的意思：tensorflow example、pose estimation、text view */
                            tvFPS.text = getString(R.string.tfe_pe_tv_fps, fps)
                        }

                        /**對檢測結果進行處理*/
                        override fun onDetectedInfo(
                            personScore: Float?,
                            poseLabels: List<Pair<String, Float>>?
                        ) {
                            tvScore.text = getString(R.string.tfe_pe_tv_score, personScore ?: 0f)

                            /** 分析目標姿態，给出提示 */
                            if (poseLabels != null && personScore != null && personScore > 0.3) {
                                missingCounter = 0
                                val sortedLabels = poseLabels.sortedByDescending { it.second }
                                when (sortedLabels[0].first) {

                                    "KIM1" -> {


                                        KIM2Counter = 0
                                        KIM3Counter = 0
                                        KIM4Counter = 0

                                        if (poseRegister == "KIM1") {
                                            KIM1Counter++
                                        }
                                        poseRegister = "KIM1"

                                        /**顯示當前姿勢狀態：kim1 */
                                        if (KIM1Counter > 60) {




                                            ivStatus.setImageResource(R.drawable.kim1)
                                        } else if (KIM1Counter > 10) {
                                            ivStatus.setImageResource(R.drawable.kim1)
                                        }

                                        /** 顯示 Debug 信息 */
                                        tvDebug.text = getString(R.string.tfe_pe_tv_debug, "${sortedLabels[0].first} $KIM1Counter")
                                    }
                                    "KIM2" -> {

                                        standardCounter = 0
                                        KIM1Counter = 0

                                        KIM3Counter = 0
                                        KIM4Counter = 0

                                        if (poseRegister == "KIM2") {
                                            KIM2Counter++
                                        }
                                        poseRegister = "KIM2"

                                        /**顯示當前姿勢狀態：KIM1 */
                                        if (KIM1Counter > 60) {



                                            ivStatus.setImageResource(R.drawable.kim2)
                                        } else if (KIM2Counter > 10) {
                                            ivStatus.setImageResource(R.drawable.kim2)
                                        }

                                        /** 顯示 Debug 信息 */
                                        tvDebug.text = getString(R.string.tfe_pe_tv_debug, "${sortedLabels[0].first} $KIM2Counter")
                                    }
                                    "KIM3" -> {

                                        standardCounter = 0
                                        KIM1Counter = 0
                                        KIM2Counter = 0

                                        KIM4Counter = 0

                                        if (poseRegister == "KIM3") {
                                            KIM3Counter++
                                        }
                                        poseRegister = "KIM3"

                                        /**顯示當前姿勢狀態：KIM3 */
                                        if (KIM3Counter > 60) {



                                            ivStatus.setImageResource(R.drawable.kim3)
                                        } else if (KIM3Counter > 10) {
                                            ivStatus.setImageResource(R.drawable.kim3)
                                        }

                                        /** 顯示 Debug 信息 */
                                        tvDebug.text = getString(R.string.tfe_pe_tv_debug, "${sortedLabels[0].first} $KIM3Counter")
                                    }
                                    "KIM4" -> {

                                        standardCounter = 0
                                        KIM1Counter = 0
                                        KIM2Counter = 0
                                        KIM3Counter = 0


                                        if (poseRegister == "KIM4") {
                                            KIM4Counter++
                                        }
                                        poseRegister = "KIM4"

                                        /**顯示當前姿勢狀態：KIM4 */
                                        if (KIM4Counter > 60) {


                                            ivStatus.setImageResource(R.drawable.kim4)
                                        } else if (KIM4Counter > 10) {
                                            ivStatus.setImageResource(R.drawable.kim4)
                                        }

                                        /** 顯示 Debug 信息 */
                                        tvDebug.text = getString(R.string.tfe_pe_tv_debug, "${sortedLabels[0].first} $KIM4Counter")
                                    }
                                    else -> {

                                        KIM1Counter = 0
                                        KIM2Counter = 0
                                        KIM3Counter = 0
                                        KIM4Counter = 0
                                        if (poseRegister == "standard") {
                                            standardCounter++
                                        }
                                        poseRegister = "standard"

                                        /** 顯示當前坐姿狀態：標準 */
                                        if (standardCounter > 20) {



                                            ivStatus.setImageResource(R.drawable.standard)
                                        }

                                        /** 顯示 Debug 信息 */
                                        tvDebug.text = getString(R.string.tfe_pe_tv_debug, "${sortedLabels[0].first} $standardCounter")
                                    }
                                }


                            }
                            else {
                                missingCounter++
                                if (missingCounter > 30) {
                                    ivStatus.setImageResource(R.drawable.no_target)
                                }

                                /** 顯示 Debug 信息 */
                                tvDebug.text = getString(R.string.tfe_pe_tv_debug, "missing $missingCounter")
                            }
                        }
                    }).apply {
                        prepareCamera()
                    }
                isPoseClassifier()
                lifecycleScope.launch(Dispatchers.Main) {
                    cameraSource?.initCamera()
                }
            }
            createPoseEstimator()
        }
    }

    private fun isPoseClassifier() {
        cameraSource?.setClassifier(if (isClassifyPose) PoseClassifier.create(this) else null)
    }

    /**初始化運計算設備選菜單（CPU、GPU、NNAPI） */
    private fun initSpinner() {
        ArrayAdapter.createFromResource(
            this,
            R.array.tfe_pe_device_name, android.R.layout.simple_spinner_item
        ).also { adapter ->
            adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)

            spnDevice.adapter = adapter
            spnDevice.onItemSelectedListener = changeDeviceListener
        }

        ArrayAdapter.createFromResource(
            this,
            R.array.tfe_pe_camera_name, android.R.layout.simple_spinner_item
        ).also { adapter ->
            adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)

            spnCamera.adapter = adapter
            spnCamera.onItemSelectedListener = changeCameraListener
        }
    }

    /** 在程序运行过程中切换运算设备 */
    private fun changeDevice(position: Int) {
        val targetDevice = when (position) {
            0 -> Device.CPU
            1 -> Device.GPU
            else -> Device.NNAPI
        }
        if (device == targetDevice) return
        device = targetDevice
        createPoseEstimator()
    }

    /**在程序運行過程中切換運行計算設備 */
    private fun changeCamera(direaction: Int) {
        val targetCamera = when (direaction) {
            0 -> Camera.BACK
            else -> Camera.FRONT
        }
        if (selectedCamera == targetCamera) return
        selectedCamera = targetCamera

        cameraSource?.close()
        cameraSource = null
        openCamera()
    }

    private fun createPoseEstimator() {
        val poseDetector = MoveNet.create(this, device, ModelType.Thunder)
        poseDetector.let { detector ->
            cameraSource?.setDetector(detector)
        }
    }

    private fun requestPermission() {
        when (PackageManager.PERMISSION_GRANTED) {
            ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.CAMERA
            ) -> {
                openCamera()
            }
            else -> {
                requestPermissionLauncher.launch(
                    Manifest.permission.CAMERA
                )
            }
        }
    }

    /**顯示報錯信息 */
    class ErrorDialog : DialogFragment() {
        override fun onCreateDialog(savedInstanceState: Bundle?): Dialog =
            AlertDialog.Builder(activity)
                .setMessage(requireArguments().getString(ARG_MESSAGE))
                .setPositiveButton(android.R.string.ok) { _, _ ->
                    // pass
                }
                .create()

        companion object {

            @JvmStatic
            private val ARG_MESSAGE = "message"

            @JvmStatic
            fun newInstance(message: String): ErrorDialog = ErrorDialog().apply {
                arguments = Bundle().apply { putString(ARG_MESSAGE, message) }
            }
        }
    }
}
