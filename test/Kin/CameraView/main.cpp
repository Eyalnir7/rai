
#include <Kin/frame.h>
#include <Kin/cameraview.h>
#include <Gui/viewer.h>
#include <Geo/depth2PointCloud.h>

//===========================================================================

void TEST(CameraView){
  rai::Configuration C;
  C.addFile("../../../../rai-robotModels/pr2/pr2.g");
  C.addFile("../../../../rai-robotModels/objects/kitchen.g");
  C.optimizeTree();

  rai::CameraView V(C, true);

  V.addSensor("kinect", "endeffKinect", 640, 480, 580./480., -1., {.1, 50.} );
//  V.selectSensor("kinect");

  Var<byteA> image;
  Var<floatA> depth;
  Var<byteA> segmentation;
  Var<arr> pts;

  PointCloudViewerCallback v(pts, image);
  ImageViewerCallback v2(image);
  ImageViewerCallback v3(segmentation);

  V.computeImageAndDepth(image.set(), depth.set());
  segmentation.set() = V.computeSegmentationImage();
  depthData2pointCloud(pts.set(), depth.get(), V.getFxycxy());

  rai::wait();

//  V.addCamera("default", "")

//  K.gl().camera.setKinect();
//  K.gl().camera.X = K.getFrameByName("endeffEyes")->X * K.gl().camera.X;
//  K.view(true); //if commented, glut/gtk is never initiated
//  byteA indexRgb, depth;
//  K.glGetMasks(580, 480);
//  write_ppm(K.gl().captureImage, "z.rgb.ppm");
//  write_ppm(convert<byte>(255.f*K.gl().captureDepth), "z.depth.ppm");

}

// =============================================================================

int MAIN(int argc,char **argv){
  rai::initCmdLine(argc, argv);

  testCameraView();

  return 0;
}
