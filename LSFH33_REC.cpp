#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <vector>
#include <string>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/pfh.h>
#include <pcl/visualization/pcl_plotter.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/recognition/cg/geometric_consistency.h>
#include <pcl/common/transforms.h>
#include <pcl/recognition/cg/hough_3d.h>
#include <pcl/features/board.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include <pcl/registration/correspondence_rejection_one_to_one.h>
#include <pcl/keypoints/uniform_sampling.h>
#include <pcl/features/pfh.h>
#include <thread>
#include <mutex>
//NOMINMAX
//_ARM64_
using namespace std;  // 可以加入 std 的命名空间

#define Feature FPFHSignature33
#define H histogram

mutex m;
struct i_p_t {
	int i = 0;
	float percent = 0.0f;
	Eigen::Matrix4f trans;
};
struct c_k_f {
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
	pcl::PointCloud<pcl::PointXYZ>::Ptr key;
	pcl::PointCloud<pcl::Feature>::Ptr feature;
};


///////////////////显示//////////////////////////////////////////////////////////////////////////////
void show_key_scene(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr key) {
	// 初始化点云可视化对象
	pcl::visualization::PCLVisualizer view("3D Viewer");
	view.setBackgroundColor(255, 255, 255);  //白色背景

	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_cloud(cloud, 0, 0, 255);//BLUE
	view.addPointCloud<pcl::PointXYZ>(cloud, color_cloud, "1");

	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_key(key, 255, 0, 0);//关键点
	view.addPointCloud<pcl::PointXYZ>(key, color_key, "2");
	view.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,6, "2");

	// 等待直到可视化窗口关闭
	while (!view.wasStopped())
	{
		view.spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
}
void show_key_model(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr key) {
	// 初始化点云可视化对象
	pcl::visualization::PCLVisualizer view("3D Viewer");
	view.setBackgroundColor(255, 255, 255);  //白色背景

	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_cloud(cloud, 0,255,0);//GREEN
	view.addPointCloud<pcl::PointXYZ>(cloud, color_cloud, "1");

	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_key(key, 255, 0, 0);//关键点
	view.addPointCloud<pcl::PointXYZ>(key, color_key, "2");
	view.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 6, "2");

	// 等待直到可视化窗口关闭
	while (!view.wasStopped())
	{
		view.spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
}

void show_scene_model_cloud(pcl::PointCloud<pcl::PointXYZ>::Ptr scene_cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr model_cloud) {
	// 初始化点云可视化对象
	pcl::visualization::PCLVisualizer view("3D Viewer");
	view.setBackgroundColor(255, 255, 255);  //白色背景

	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_source(scene_cloud, 0, 255, 0);//绿色点云
	view.addPointCloud<pcl::PointXYZ>(scene_cloud, color_source, "1");

	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_target(model_cloud, 0, 0, 255);//蓝色点云
	view.addPointCloud<pcl::PointXYZ>(model_cloud, color_target, "2");

	// 等待直到可视化窗口关闭
	while (!view.wasStopped())
	{
		view.spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
}

void show_point_cloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
	// 初始化点云可视化对象
	pcl::visualization::PCLVisualizer view("3D Viewer");
	view.setBackgroundColor(255, 255, 255); //白色背景
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_cloud(cloud, 0, 255, 0);//绿色点云
	view.addPointCloud<pcl::PointXYZ>(cloud, color_cloud, "1");
	// 等待直到可视化窗口关闭
	while (!view.wasStopped())
	{
		view.spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
}

void show_point_clouds(std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& clouds) {
	// 初始\化点云可视化对象
	pcl::visualization::PCLVisualizer viewer_final("3D Viewer");
	viewer_final.setBackgroundColor(255, 255, 255);   //白色背景
	for (int i = 0; i < clouds.size(); i++) {
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color(clouds[i], rand() % 255, rand() % 255, rand() % 255);
		viewer_final.addPointCloud<pcl::PointXYZ>(clouds[i], color, std::to_string(i));
	}
	// 等待直到可视化窗口关闭
	while (!viewer_final.wasStopped())
	{
		viewer_final.spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
}

void show_coor(pcl::PointCloud<pcl::PointXYZ>::Ptr model_cloud, pcl::PointCloud<pcl::PointXYZ> model_key,
	pcl::PointCloud<pcl::Feature>::Ptr model_feature,
	pcl::PointCloud<pcl::PointXYZ>::Ptr scene_cloud, pcl::PointCloud<pcl::PointXYZ> scene_key,
	pcl::PointCloud<pcl::Feature>::Ptr scene_feature,
	pcl::CorrespondencesPtr& corr) {

	for (int i = 0; i < corr->size(); i++) {
		cout << corr->at(i).index_query << "---" << corr->at(i).index_match << "---" << corr->at(i).distance << endl;
		pcl::visualization::PCLPlotter plotter;
		plotter.addFeatureHistogram<pcl::Feature>(*model_feature, "fpfh", corr->at(i).index_query);
		plotter.addFeatureHistogram<pcl::Feature>(*scene_feature, "fpfh", corr->at(i).index_match);
		std::cout << model_feature->points[corr->at(i).index_query] << endl;
		std::cout << scene_feature->points[corr->at(i).index_match] << endl;
		pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints_ptr_model(new pcl::PointCloud<pcl::PointXYZ>());
		pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints_ptr_scenes(new pcl::PointCloud<pcl::PointXYZ>());
		keypoints_ptr_model->push_back(model_key.points[corr->at(i).index_query]);
		keypoints_ptr_scenes->push_back(scene_key.points[corr->at(i).index_match]);
		boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));

		int v1(0);
		viewer->createViewPort(0.0, 0.0, 0.5, 1.0, v1);  //4个参数分别是X轴的最小值，最大值，Y轴的最小值，最大值，取值0-1，v1是标识
		viewer->setBackgroundColor(255, 255, 255, v1);    //设置视口的背景颜色
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_key_model(keypoints_ptr_model, 255, 0, 0);
		viewer->addPointCloud<pcl::PointXYZ>(keypoints_ptr_model, color_key_model, "color_key_model", v1);
		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "color_key_model");
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_cloud_model(model_cloud, 0, 255, 0);
		viewer->addPointCloud<pcl::PointXYZ>(model_cloud, color_cloud_model, "cloud_model", v1);
		int v2(0);
		viewer->createViewPort(0.5, 0.0, 1.0, 1.0, v2);  //4个参数分别是X轴的最小值，最大值，Y轴的最小值，最大值，取值0-1，v1是标识
		viewer->setBackgroundColor(255, 255, 255, v2);    //设置视口的背景颜色
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_key_scenes(keypoints_ptr_scenes, 255, 0, 0);
		viewer->addPointCloud<pcl::PointXYZ>(keypoints_ptr_scenes, color_key_scenes, "color_key_scenes", v2);
		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "color_key_scenes");
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_cloud_scenes(scene_cloud, 0, 0, 255);
		viewer->addPointCloud<pcl::PointXYZ>(scene_cloud, color_cloud_scenes, "cloud_scenes", v2);

		//plotter.plot();
		while (!viewer->wasStopped())
		{
			viewer->spinOnce(100);
			boost::this_thread::sleep(boost::posix_time::microseconds(100000));
		}
	}
}

void show_line(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_target,
	pcl::PointCloud<pcl::PointXYZ>::Ptr key_source, pcl::PointCloud<pcl::PointXYZ>::Ptr key_target,
	pcl::CorrespondencesPtr& corr, float& leaf_size) {

	pcl::PointCloud<pcl::PointXYZ>::Ptr new_key_source(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr new_key_target(new pcl::PointCloud<pcl::PointXYZ>());
	for (int i = 0; i < corr->size(); i++) {
		new_key_source->push_back(key_source->points[corr->at(i).index_query]);
		new_key_target->push_back(key_target->points[corr->at(i).index_match]);
	}
	pcl::PointCloud<pcl::PointXYZ>::Ptr new_cloud_source(new pcl::PointCloud<pcl::PointXYZ>);
	*new_cloud_source = *cloud_source;
	for (int i = 0; i < cloud_source->size(); i++) {
		new_cloud_source->points[i].x += 300.0f* leaf_size;
	}
	for (int i = 0; i < new_key_source->size(); i++) {
		new_key_source->points[i].x += 300.0f* leaf_size;
	}
	pcl::visualization::PCLVisualizer line("line");
	line.setBackgroundColor(255, 255, 255);
	line.addPointCloud(cloud_target, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(cloud_target, 0, 255, 0), "cloud_target");
	line.addPointCloud(new_cloud_source, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(new_cloud_source, 0, 0, 255), "cloud_source");
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>color_new_key_target(new_key_target, 255, 0, 0);
	line.addPointCloud<pcl::PointXYZ>(new_key_target, color_new_key_target, "new_key_target");
	line.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "new_key_target");
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_new_key_source(new_key_source, 255, 0, 0);
	line.addPointCloud<pcl::PointXYZ>(new_key_source, color_new_key_source, "new_key_source");
	line.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "new_key_source");

	for (int i = 0; i < new_key_source->size(); i++)
	{
		pcl::PointXYZ source_point = new_key_source->points[i];
		pcl::PointXYZ target_point = new_key_target->points[i];
		line.addLine(source_point, target_point, 0, 0, 0, std::to_string(i));
		line.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 2, std::to_string(i));
	}
	line.spin();
}

void show_line(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source, pcl::PointCloud<pcl::PointXYZ>::Ptr key_source,
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_target, pcl::PointCloud<pcl::PointXYZ>::Ptr key_target,
	std::vector<int>& corr, float& leaf_size) {

	//////////////////将点云平移，方便显示////////////////////////////////////////////////////////////////////////
	pcl::PointCloud<pcl::PointXYZ>::Ptr new_key_source(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr new_key_target(new pcl::PointCloud<pcl::PointXYZ>);
	for (int i = 0; i < corr.size(); i++) {
		new_key_source->push_back(key_source->points[corr[i]]);
		new_key_target->push_back(key_target->points[corr[i]]);
	}
	pcl::PointCloud<pcl::PointXYZ>::Ptr new_cloud_source(new pcl::PointCloud<pcl::PointXYZ>);
	*new_cloud_source = *cloud_source;
	for (int i = 0; i < new_cloud_source->size(); i++) {
		new_cloud_source->points[i].x += 300.0f* leaf_size;
		//new_cloud_source->points[i].y += 300.0f* leaf_size;
	}
	for (int i = 0; i < new_key_source->size(); i++) {
		new_key_source->points[i].x += 300.0f* leaf_size;
		//new_key_source->points[i].y += 300.0f* leaf_size;
	}
	////////////////////显示对应点连线//////////////////////////////////////////////////////////////////////
	pcl::visualization::PCLVisualizer line("line");
	line.setBackgroundColor(255, 255, 255);
	line.addPointCloud(new_cloud_source, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(new_cloud_source, 0, 0, 255), "new_cloud_source");
	line.addPointCloud(cloud_target, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(cloud_target, 0, 255, 0), "cloud_target");

	line.addPointCloud<pcl::PointXYZ>(new_key_source, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(new_key_source, 255, 0, 0), "new_key_source");
	line.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "new_key_source");

	line.addPointCloud<pcl::PointXYZ>(new_key_target, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(new_key_target, 255, 0, 0), "new_key_target");
	line.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "new_key_target");

	for (int i = 0; i < new_key_source->size(); i++)
	{
		line.addLine(new_key_source->points[i], new_key_target->points[i], 0, 0, 0, std::to_string(i));
		line.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 2, std::to_string(i));
	}
	line.spin();
}

void show_key_corr(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source, pcl::PointCloud<pcl::PointXYZ>::Ptr key_source,
	pcl::PointXYZ ps0, pcl::PointXYZ ps1, pcl::PointXYZ ps2,
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_target, pcl::PointCloud<pcl::PointXYZ>::Ptr key_target,
	pcl::PointXYZ pt0, pcl::PointXYZ pt1, pcl::PointXYZ pt2) {

	pcl::PointCloud<pcl::PointXYZ>::Ptr key_ps0(new pcl::PointCloud<pcl::PointXYZ>);
	key_ps0->push_back(ps0);
	pcl::PointCloud<pcl::PointXYZ>::Ptr key_ps(new pcl::PointCloud<pcl::PointXYZ>);
	key_ps->push_back(ps1);
	key_ps->push_back(ps2);
	pcl::PointCloud<pcl::PointXYZ>::Ptr key_pt0(new pcl::PointCloud<pcl::PointXYZ>);
	key_pt0->push_back(pt0);
	pcl::PointCloud<pcl::PointXYZ>::Ptr key_pt(new pcl::PointCloud<pcl::PointXYZ>);
	key_pt->push_back(pt1);
	key_pt->push_back(pt2);
	pcl::visualization::PCLVisualizer view("3D Viewer");
	int v1(0);
	int v2(1);
	view.createViewPort(0.0, 0.0, 0.5, 1.0, v1);
	view.createViewPort(0.5, 0.0, 1.0, 1.0, v2);
	view.setBackgroundColor(255, 255, 255, v1);
	view.setBackgroundColor(255, 255, 255, v2);
	view.addPointCloud<pcl::PointXYZ>(cloud_source, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(cloud_source, 0, 0, 0), "cloud_source", v1);
	view.addPointCloud<pcl::PointXYZ>(key_ps0, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(key_ps0, 255, 0, 0), "key_ps0", v1);
	view.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "key_ps0", v1);
	view.addPointCloud<pcl::PointXYZ>(key_ps, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(key_ps, 0, 255, 0), "key_ps", v1);
	view.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "key_ps", v1);
	view.addPointCloud<pcl::PointXYZ>(cloud_target, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(cloud_target, 0, 0, 0), "cloud_target", v2);
	view.addPointCloud<pcl::PointXYZ>(key_pt0, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(key_pt0, 255, 0, 0), "key_pt0", v2);
	view.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "key_pt0", v2);
	view.addPointCloud<pcl::PointXYZ>(key_pt, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(key_pt, 0, 255, 0), "key_pt", v2);
	view.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "key_pt", v2);
	while (!view.wasStopped()) {
		view.spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
}

void show_point_clouds_and_trans_models(std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> scenes,
	std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> models, std::vector<i_p_t> result_final) {

	vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> models_results;
	pcl::visualization::PCLVisualizer viewer_final("3D Viewer");
	viewer_final.setBackgroundColor(255, 255, 255);   //白色背景
	for (int i = 0; i < scenes.size(); i++) {
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color(scenes[i], rand() % 255, rand() % 255, rand() % 255);
		viewer_final.addPointCloud<pcl::PointXYZ>(scenes[i], color, "scenes" + to_string(i));
		viewer_final.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "scenes" + to_string(i));
		pcl::PointCloud<pcl::PointXYZ>::Ptr models_result(new pcl::PointCloud<pcl::PointXYZ>);
		*models_result = *models[result_final[i].i];
		pcl::transformPointCloud(*models_result, *models_result, result_final[i].trans);
		models_results.push_back(models_result);
		if (result_final[i].percent < 0.8f)
			continue;
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_model(models_results[i], rand() % 255, rand() % 255, rand() % 255);
		viewer_final.addPointCloud<pcl::PointXYZ>(models_results[i], color_model, "models_results" + to_string(i));
	}
	while (!viewer_final.wasStopped()) {
		viewer_final.spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
}


/////////////////循环滤波///////////////////////////////////////////////////////////////////////////////////////////////////////
pcl::PointCloud<pcl::Normal>::Ptr normal_estimation_OMP(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float radius) {
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>());
	pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;
	ne.setNumberOfThreads(10);
	ne.setInputCloud(cloud);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
	ne.setSearchMethod(tree);
	ne.setRadiusSearch(radius);
	//ne.setKSearch(k);
	ne.compute(*normals);
	return normals;
}

float com_leaf(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	kdtree.setInputCloud(cloud);//Acloud在Bcloud中进行搜索
	int K = 2;
	std::vector<int> pointIdxNKNSearch(K);//最近点索引
	std::vector<float> pointNKNSquaredDistance(K);//最近点距离
	float leaf_size = 0;
	for (int i = 0; i < cloud->size(); i++) {
		kdtree.nearestKSearch(cloud->points[i], K, pointIdxNKNSearch, pointNKNSquaredDistance);
		leaf_size = leaf_size + sqrt(pointNKNSquaredDistance[1]);
		pointIdxNKNSearch.clear();
		pointNKNSquaredDistance.clear();
	}
	leaf_size = (float)leaf_size / (float)(cloud->size());
	//std::cout << "平均距离：" << leaf_size << "点云点数：" << cloud->size() << std::endl;
	return leaf_size;
}

float get_leaf_size_by_leaf_size(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float leaf_size, float target_leaf_size) {
	return target_leaf_size + 0.2f*(target_leaf_size - leaf_size);
}

pcl::PointCloud<pcl::PointXYZ>::Ptr voxel_grid(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float leaf_size) {
	//体素滤波
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::VoxelGrid<pcl::PointXYZ> sor;  //创建滤波对象
	sor.setInputCloud(cloud);            //设置需要过滤的点云给滤波对象
	sor.setLeafSize(leaf_size, leaf_size, leaf_size);  //设置滤波时创建的体素体积
	sor.filter(*cloud_filtered);           //执行滤波处理，存储输出	
	return cloud_filtered;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr filter_to_leaf_size(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float target_leaf_size) {
	float leaf_size = com_leaf(cloud);
	if (target_leaf_size > leaf_size) {
		*cloud = *voxel_grid(cloud, target_leaf_size);
	}
	else {
		return cloud;
	}
	leaf_size = com_leaf(cloud);
	int num = 0;
	while (target_leaf_size > leaf_size + leaf_size * 0.02f) {
		num = cloud->size();
		leaf_size = get_leaf_size_by_leaf_size(cloud, leaf_size, target_leaf_size);
		*cloud = *voxel_grid(cloud, leaf_size);
		leaf_size = com_leaf(cloud);
		if (cloud->size() == num) {
			break;
		}
	}
	return cloud;
}

///////////////////////关键点查找//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
float com_avg_curvature(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::Normal>::Ptr normals, int i,
	float r_avg, pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr tree) {
	float avg_curvature = 0;
	std::vector<int> point_ind;
	std::vector<float> point_dist;
	tree->radiusSearch(cloud->points[i], r_avg, point_ind, point_dist);
	for (int i = 0; i < point_ind.size(); i++) {
		avg_curvature += normals->points[point_ind[i]].curvature;
	}
	avg_curvature = avg_curvature / float(point_ind.size());
	return avg_curvature;
}

bool is_max_avg_curvature(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, std::vector<float> avg_c, int i,
	std::vector<bool>& pre_key,
	pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr tree, float r_max) {

	std::vector<int> id;
	std::vector<float> dis;
	tree->radiusSearch(cloud->points[i], r_max, id, dis);//此处半径为计算曲率和的半径
	if (id.size() < 5)
		return false;
	for (int i = 1; i < id.size(); i++) {
		if (pre_key[id[i]]) {
			if (avg_c[id[0]] > avg_c[id[i]])
				pre_key[id[i]] = false;
			else if (avg_c[id[0]] < avg_c[id[i]])
				pre_key[id[0]] = false;
		}
	}
	return pre_key[id[0]];
}

pcl::PointCloud<pcl::PointXYZ>::Ptr key_detect(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
	pcl::PointCloud<pcl::Normal>::Ptr normal,
	float r_avg, float r_max) {
	pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr tree(new pcl::KdTreeFLANN<pcl::PointXYZ>);
	tree->setInputCloud(cloud);

	std::vector<bool> pre_key(cloud->size(), false);
	for (int i = 0; i < cloud->size(); i++) {
		if (normal->points[i].curvature > 0.01) {
			pre_key[i] = true;
		}
	}
	std::vector<bool> possible_key_possible(pre_key);
	std::vector<float> avg_c;
	for (int i = 0; i < cloud->size(); i++) {
		if (pre_key[i])
			avg_c.push_back(com_avg_curvature(cloud, normal, i, r_avg, tree));
		else
			avg_c.push_back(0);
	}
	pcl::PointCloud<pcl::PointXYZ>::Ptr key(new pcl::PointCloud<pcl::PointXYZ>);
	for (int i = 0; i < cloud->size(); i++) {
		if (pre_key[i]) {
			if (is_max_avg_curvature(cloud, avg_c, i, pre_key, tree, r_max)) {//此处半径为计算曲率和的半径
				key->push_back(cloud->points[i]);
			}
		}
	}
	return key;
}


/////////////////////特征描述/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

float com_angle(float nx, float ny, float nz, float cx, float cy, float cz) {
	if ((cx == 0 && cy == 0 && cz == 0) || isnan(nx) || isnan(ny) || isnan(nz))
		return 0;
	float angle = acos((nx*cx + ny * cy + nz * cz) / (sqrt(pow(nx, 2) + pow(ny, 2) + pow(nz, 2))*sqrt(pow(cx, 2) + pow(cy, 2) + pow(cz, 2))))*(180.0f / 3.1415f);
	return angle;
}

float com_dis(float nx, float ny, float nz, float cx, float cy, float cz) {
	float distance = 0;
	distance = sqrt(pow(nx - cx, 2) + pow(ny - cy, 2) + pow(nz - cz, 2));
	return distance;
}

pcl::PointCloud<pcl::Feature>::Ptr com_lsfh33_feature(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
	pcl::PointCloud<pcl::Normal>::Ptr normal, pcl::PointCloud<pcl::PointXYZ>::Ptr key,
	float r, int num_distance = 15, int num_angle = 18) {

	pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr tree(new pcl::KdTreeFLANN<pcl::PointXYZ>);
	tree->setInputCloud(cloud);
	pcl::PointCloud<pcl::Feature>::Ptr features(new pcl::PointCloud<pcl::Feature>());
	for (int k = 0; k < key->size(); k++) {
		std::vector<int> indices;
		std::vector<float> dists;
		tree->radiusSearch(key->points[k], r, indices, dists);
		pcl::Feature feature;
		for (int i = 0; i < feature.descriptorSize(); i++) {
			feature.histogram[i] = 0;
		}
		pcl::PointCloud<pcl::PointXYZ>::Ptr search_points(new pcl::PointCloud<pcl::PointXYZ>());
		pcl::PointCloud<pcl::Normal>::Ptr search_normals(new pcl::PointCloud<pcl::Normal>());
		pcl::copyPointCloud(*cloud, indices, *search_points);
		pcl::copyPointCloud(*normal, indices, *search_normals);
		Eigen::Vector4f centroid;  //质心 
		pcl::compute3DCentroid(*search_points, centroid); //估计质心的坐标
		//cout << centroid.x() << " " << centroid.y() << " " << centroid.z() << endl;
		std::vector<float> vec_distance;
		for (int i = 0; i < search_points->size(); i++) {
			float distance = 0;
			distance = com_dis(search_points->points[i].x, search_points->points[i].y, search_points->points[i].z, centroid.x(), centroid.y(), centroid.z());
			vec_distance.push_back(distance);
		}
		float max_diatance = *std::max_element(std::begin(vec_distance), std::end(vec_distance));
		float min_diatance = *std::min_element(std::begin(vec_distance), std::end(vec_distance));
		float res_distance = (max_diatance - min_diatance) / num_distance;
		for (int i = 0; i < search_points->size(); i++) {
			float angle = com_angle(search_normals->points[i].normal_x, search_normals->points[i].normal_y, search_normals->points[i].normal_z,
				search_points->points[i].x - centroid.x(), search_points->points[i].y - centroid.y(), search_points->points[i].z - centroid.z());
			int bin_angle = int(angle / 10.0f);
			int bin_distance = 0;
			if (res_distance != 0) {
				bin_distance = int((vec_distance[i] - min_diatance) / res_distance);
			}
			if (bin_distance > num_distance - 1) bin_distance = num_distance - 1;
			if (bin_angle > num_angle - 1) bin_angle = num_angle - 1;
			//feature.histogram[bin_distance] += 1;
			feature.histogram[bin_distance] += 1;
			feature.histogram[num_distance + bin_angle] += 1;
		}
		for (int i = 0; i < feature.descriptorSize(); i++) {
			feature.histogram[i] = feature.histogram[i] / ((float)search_points->size());
		}
		features->push_back(feature);
		//pcl::visualization::PCLPlotter plotter;
		//plotter.addFeatureHistogram<pcl::FPFHSignature33>(*features, "fpfh", k);
		//plotter.plot();
	}
	return features;
}


/////////////////////特征匹配/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
float com_angle(pcl::PointXYZ p, pcl::PointXYZ p1, pcl::PointXYZ p2) {
	return acos(((p1.x - p.x)*(p2.x - p.x) + (p1.y - p.y)*(p2.y - p.y) + (p1.z - p.z)*(p2.z - p.z)) / (sqrt(pow(p1.x - p.x, 2) + pow(p1.y - p.y, 2) + pow(p1.z - p.z, 2))*sqrt(pow(p2.x - p.x, 2) + pow(p2.y - p.y, 2) + pow(p2.z - p.z, 2))))*(180.0f / 3.1415f);
}

float com_feature_dist(pcl::Feature f1, pcl::Feature f2) {
	float dist = 0.0f;
	for (int i = 0; i < f1.descriptorSize(); i++) {
		dist += pow(f1.H[i] - f2.H[i], 2);
		//dist += pow(f1.histogram[i] - f2.histogram[i], 2);
	}
	return sqrt(dist);
}

pcl::CorrespondencesPtr com_corr3(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source, pcl::PointCloud<pcl::PointXYZ>::Ptr key_source,
	pcl::PointCloud<pcl::Feature>::Ptr feature_source, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_target,
	pcl::PointCloud<pcl::PointXYZ>::Ptr key_target, pcl::PointCloud<pcl::Feature>::Ptr feature_target,
	float dis_f, float dis_k, float dis_a) {

	pcl::CorrespondencesPtr corrs(new pcl::Correspondences());
	pcl::KdTreeFLANN<pcl::Feature> feature_target_kdtree;
	feature_target_kdtree.setInputCloud(feature_target);
	pcl::KdTreeFLANN<pcl::PointXYZ> key_source_kdtree;
	key_source_kdtree.setInputCloud(key_source);
	pcl::KdTreeFLANN<pcl::PointXYZ> key_target_kdtree;
	key_target_kdtree.setInputCloud(key_target);
	for (size_t i = 0; i < key_source->size(); i++) {
		std::vector<int> key_source_indices;
		std::vector<float> key_source_dists;
		key_source_kdtree.nearestKSearch(key_source->points[i], 3, key_source_indices, key_source_dists);
		float angle1 = com_angle(key_source->points[i], key_source->points[key_source_indices[1]], key_source->points[key_source_indices[2]]);
		std::vector<int> corr_indices;
		std::vector<float> corr_dists;
		feature_target_kdtree.nearestKSearch(feature_source->points[i], 10, corr_indices, corr_dists);
		if (sqrt(corr_dists[0]) < 0.5*sqrt(corr_dists[1])) {
			pcl::Correspondence corr0(i, corr_indices[0], sqrt(corr_dists[0]));
			corrs->push_back(corr0);
			continue;
		}
		for (int j = 0; j < corr_indices.size(); j++) {
			if (sqrt(corr_dists[j]) < dis_f) {
				std::vector<int> key_target_indices;
				std::vector<float> key_target_dists;
				key_target_kdtree.nearestKSearch(key_target->points[corr_indices[j]], 3, key_target_indices, key_target_dists);
				float angle2 = com_angle(key_target->points[corr_indices[j]], key_target->points[key_target_indices[1]], key_target->points[key_target_indices[2]]);
				float dis_f1 = com_feature_dist(feature_source->points[key_source_indices[1]], feature_target->points[key_target_indices[1]]);
				float dis_f2 = com_feature_dist(feature_source->points[key_source_indices[2]], feature_target->points[key_target_indices[2]]);
				float dis_k1 = abs(sqrt(key_source_dists[1]) - sqrt(key_target_dists[1]));
				float dis_k2 = abs(sqrt(key_source_dists[2]) - sqrt(key_target_dists[2]));
				float dis_angle = abs(angle1 - angle2);
				//cout << "dist1:" << dist1 << endl;
				//cout << "dist2:" << dist2 << endl;
				//cout << "feature_dist0:" << sqrt(corr_dists[j]) << endl;
				//cout << "feature_dist1:" << feature_dist1 << endl;
				//cout << "feature_dist2:" << feature_dist2 << endl;
				//cout << "dist_angle:" << dist_angle << endl;
				/*show_key_corr(cloud_source, key_source,
					key_source->points[key_source_indices[0]], key_source->points[key_source_indices[1]], key_source->points[key_source_indices[2]],
					cloud_target, key_target,
					key_target->points[key_target_indices[0]], key_target->points[key_target_indices[1]], key_target->points[key_target_indices[2]]);*/
				if (dis_k1 < dis_k&&dis_k2 < dis_k&&dis_f1 < dis_f&&dis_f2 < dis_f&&dis_angle < dis_a) {
					//show_key_corr(cloud_source, key_source,
					//	key_source->points[key_source_indices[0]], key_source->points[key_source_indices[1]], key_source->points[key_source_indices[2]],
					//	cloud_target, key_target,
					//	key_target->points[key_target_indices[0]], key_target->points[key_target_indices[1]], key_target->points[key_target_indices[2]]);
					pcl::Correspondence corr(i, corr_indices[j], sqrt(corr_dists[j]));
					corrs->push_back(corr);
					pcl::Correspondence corr1(key_source_indices[1], key_target_indices[1], dis_f1);
					corrs->push_back(corr1);
					pcl::Correspondence corr2(key_source_indices[2], key_target_indices[2], dis_f2);
					corrs->push_back(corr2);
					//break;
				}
			}
			else break;
		}
	}
	return corrs;
}

///////////////////点云加噪声////////////////////////////////////
pcl::PointCloud<pcl::PointXYZ>::Ptr add_gaussian_noise(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float m) {
	float leaf_size = 0;
	leaf_size = com_leaf(cloud);
	//添加高斯噪声
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloudfiltered(new pcl::PointCloud<pcl::PointXYZ>());
	cloudfiltered->points.resize(cloud->points.size());//将点云的cloud的size赋值给噪声
	cloudfiltered->header = cloud->header;
	cloudfiltered->width = cloud->width;
	cloudfiltered->height = cloud->height;
	boost::mt19937 rng;
	rng.seed(static_cast<unsigned int>(time(0)));
	boost::normal_distribution<> nd(0, m*leaf_size);
	boost::variate_generator<boost::mt19937&, boost::normal_distribution<>> var_nor(rng, nd);
	//添加噪声
	for (size_t point_i = 0; point_i < cloud->points.size(); ++point_i)
	{
		//cloudfiltered->points[point_i].x = cloud->points[point_i].x + static_cast<float> (var_nor());
		//cloudfiltered->points[point_i].y = cloud->points[point_i].y + static_cast<float> (var_nor());
		//cloudfiltered->points[point_i].z = cloud->points[point_i].z + static_cast<float> (var_nor());
		cloudfiltered->points[point_i].x = cloud->points[point_i].x + static_cast<float> (var_nor());
		cloudfiltered->points[point_i].y = cloud->points[point_i].y;
		cloudfiltered->points[point_i].z = cloud->points[point_i].z;
	}
	return cloudfiltered;
}

//////////////////欧式分割//////////////////////////
vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> euclidean_cluster(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
	int tolerance = 4, int min = 1000, int max = 50000) {
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
	tree->setInputCloud(cloud);
	std::vector<pcl::PointIndices> cluster_indices;
	pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;   //欧式聚类对象
	ec.setClusterTolerance(tolerance);                     // 设置近邻搜索的搜索半径为2cm
	ec.setMinClusterSize(min);                 //设置一个聚类需要的最少的点数目为100
	ec.setMaxClusterSize(max);               //设置一个聚类需要的最大点数目为25000
	ec.setSearchMethod(tree);                    //设置点云的搜索机制
	ec.setInputCloud(cloud);
	ec.extract(cluster_indices);           //从点云中提取聚类，并将点云索引保存在cluster_indices中
	vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clouds;
	for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it) {
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>);
		for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit)
			cloud_cluster->points.push_back(cloud->points[*pit]);
		cloud_cluster->width = cloud_cluster->points.size();
		cloud_cluster->height = 1;
		cloud_cluster->is_dense = true;
		clouds.push_back(cloud_cluster);
	}
	return clouds;
}

//////////////////计算重叠率////////////////////////
float com_overlap_rate(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_target, float leaf_size) {
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	kdtree.setInputCloud(cloud_target);
	int count = 0;//重合点个数
	for (int i = 0; i < cloud_source->size(); i++) {
		std::vector<int> indices;
		std::vector<float> dists;
		if (kdtree.nearestKSearch(cloud_source->points[i], 1, indices, dists) > 0) {
			if (sqrt(dists[0]) < 5.0f*leaf_size)
				count++;
		}
	}
	float overlap_rate = (float)count / (float)cloud_source->size();
	//cout << "重叠率：" << overlap_rate * 100.0f << "%" << endl;
	return overlap_rate;
}

//////////////////随机采样一致性///////////////////////////
std::vector<int> ransac(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source, pcl::PointCloud<pcl::PointXYZ>::Ptr key_source,
	pcl::PointCloud<pcl::Feature>::Ptr feature_source,
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_target, pcl::PointCloud<pcl::PointXYZ>::Ptr key_target,
	pcl::PointCloud<pcl::Feature>::Ptr feature_target,
	float leaf_size, Eigen::Matrix4f& trans) {

	////////////////////////随机采样一致性//////////////////////////////////////////////////////////////////////
	pcl::SampleConsensusPrerejective<pcl::PointXYZ, pcl::PointXYZ, pcl::Feature> align;
	align.setInputSource(key_source);
	align.setSourceFeatures(feature_source);
	align.setInputTarget(key_target);
	align.setTargetFeatures(feature_target);
	align.setMaximumIterations(1000); // Number of RANSAC iterations
	align.setNumberOfSamples(3); // Number of points to sample for generating/prerejecting a pose
	align.setCorrespondenceRandomness(3); // Number of nearest features to use
	align.setSimilarityThreshold(0.9f); // Polygonal edge length similarity threshold
	align.setMaxCorrespondenceDistance(2.5f*leaf_size); // Inlier threshold
	//align.setRANSACOutlierRejectionThreshold(5.0f * leaf_size);
	align.setInlierFraction(0.25f); // Required inlier fraction for accepting a pose hypothesis
	align.align(*key_source);
	//std::cout << "分数： " << align.getFitnessScore(5.0f*leaf_size) << std::endl;
	trans = align.getFinalTransformation();
	pcl::transformPointCloud(*cloud_source, *cloud_source, trans);
	std::vector<int> corr;
	corr = align.getInliers();
	return corr;

}

///////////////////ICP/////////////////////////////////////////////
void my_icp(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_target, float& leaf_size, Eigen::Matrix4f& trans) {

	pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
	icp.setInputSource(cloud_source);
	icp.setInputTarget(cloud_target);
	icp.setTransformationEpsilon(0.1*leaf_size);
	icp.setMaxCorrespondenceDistance(5.0f * leaf_size);
	icp.setMaximumIterations(300);
	icp.align(*cloud_source);
	//std::cout << "icp分数： " << icp.getFitnessScore(1.0f*leaf_size) << std::endl;
	Eigen::Matrix4f TR = icp.getFinalTransformation();
	//pcl::transformPointCloud(*cloud_source, *cloud_source, TR);
	trans = trans * TR;
}


//////////////////配准过程//////////////////////////////////////////////////////////////////
vector<i_p_t> results;

bool cmp(i_p_t a, i_p_t b) {
	return a.percent > b.percent;
}

void my_align(pcl::PointCloud<pcl::PointXYZ>::Ptr scene_cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr scene_key,
	pcl::PointCloud<pcl::Feature>::Ptr scene_feature,
	pcl::PointCloud<pcl::PointXYZ>::Ptr model_cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr model_key,
	pcl::PointCloud<pcl::Feature>::Ptr model_feature,
	float leaf_size, int i = 0) {

	pcl::PointCloud<pcl::PointXYZ>::Ptr scene_cloud0(new pcl::PointCloud<pcl::PointXYZ>);
	*scene_cloud0 = *scene_cloud;
	pcl::PointCloud<pcl::PointXYZ>::Ptr scene_key0(new pcl::PointCloud<pcl::PointXYZ>);
	*scene_key0 = *scene_key;
	pcl::PointCloud<pcl::Feature>::Ptr scene_feature0(new pcl::PointCloud<pcl::Feature>);
	*scene_feature0 = *scene_feature;
	i_p_t result;
	result.i = i;
	////////////////////初始对应关系估计////////////////////////////////////////////////////////////////////////
	//double start = 0, end = 0;
	//start = GetTickCount();
	pcl::CorrespondencesPtr first_corr(new pcl::Correspondences());
	*first_corr = *com_corr3(scene_cloud0, scene_key0, scene_feature0, model_cloud, model_key, model_feature, 0.2f, 5.0f*leaf_size, 30);
	show_line(scene_cloud0, model_cloud, scene_key0, model_key, first_corr, leaf_size);
	//end = GetTickCount();
	//show_coor(cloud_source, cloud_target, *key_source, *key_target, features_source, features_target, corr);
	//cout << "初始对应关系数目：" << corr->size() << endl;
	//cout << "初始对应关系估计：" << end - start << "ms" << endl;
	//show_line(cloud_source, cloud_target, key_source, key_target, corr, leaf_size);
	if (first_corr->size() < 10) {
		results.push_back(result);
		return;
	}
	pcl::registration::CorrespondenceRejectorOneToOne corr_est;
	corr_est.setInputCorrespondences(first_corr);
	corr_est.getRemainingCorrespondences(*first_corr, *first_corr);
	if (first_corr->size() < 10) {
		results.push_back(result);
		return;
	}

	/////////////////////提取初始对应关系关键点和特征///////////////////////////////////////////////////////////////////////
	//start = GetTickCount();
	pcl::PointCloud<pcl::PointXYZ>::Ptr scene_key_in_first_corr(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr model_key_in_first_corr(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::Feature>::Ptr scene_feature_in_first_corr(new pcl::PointCloud<pcl::Feature>);
	pcl::PointCloud<pcl::Feature>::Ptr model_feature_in_first_corr(new pcl::PointCloud<pcl::Feature>);
	for (int i = 0; i < first_corr->size(); i++) {
		scene_key_in_first_corr->push_back(scene_key0->points[first_corr->at(i).index_query]);
		model_key_in_first_corr->push_back(model_key->points[first_corr->at(i).index_match]);
		scene_feature_in_first_corr->push_back(scene_feature0->points[first_corr->at(i).index_query]);
		model_feature_in_first_corr->push_back(model_feature->points[first_corr->at(i).index_match]);
	}
	vector<int> final_corr;
	final_corr = ransac(scene_cloud0, scene_key_in_first_corr, scene_feature_in_first_corr,
		model_cloud, model_key_in_first_corr, model_feature_in_first_corr, leaf_size, result.trans);
	show_line(scene_cloud0, scene_key_in_first_corr, model_cloud, model_key_in_first_corr, final_corr, leaf_size);
	/*end = GetTickCount();
	cout << "ransac corr：" << end - start << "ms" << endl;

	start = GetTickCount();*/
	pcl::PointCloud<pcl::PointXYZ>::Ptr scene_cloud0_filter(new pcl::PointCloud<pcl::PointXYZ>);
	*scene_cloud0_filter = *voxel_grid(scene_cloud0, 10 * leaf_size);
	my_icp(scene_cloud0_filter, model_cloud, leaf_size, result.trans);
	//end = GetTickCount();
	//cout << "icp：" << end - start << "ms" << endl;
	//start = GetTickCount();
	result.percent = com_overlap_rate(scene_cloud0_filter, model_cloud, 5.0f*leaf_size);
	result.trans = result.trans.inverse().eval();
	m.lock();
	results.push_back(result);
	m.unlock();
	//end = GetTickCount();
	//cout << "com_overlap_rate：" << end - start << "ms" << endl;
	return;
}



i_p_t predict(pcl::PointCloud<pcl::PointXYZ>::Ptr scene_cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr scene_key,
	pcl::PointCloud<pcl::Feature>::Ptr scene_feature,
	vector<string> names, vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> model_clouds,
	vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> model_keys,
	vector<pcl::PointCloud<pcl::Feature>::Ptr> model_features, float leaf_size = 1) {

	//double start = 0;
	//double end = 0;
	//start = GetTickCount();
	results.clear();
	for (int i = 0; i < names.size(); i++) {
		thread t(my_align, scene_cloud, scene_key, scene_feature, model_clouds[i], model_keys[i], model_features[i], leaf_size, i);
		t.detach();
	}
	while (results.size() != names.size()) {
		Sleep(10);
	}
	//end = GetTickCount();
	//cout << "识别时间：" << end - start << "ms" << endl;
	sort(results.begin(), results.end(), cmp);
	return results[0];
}

///////////////////场景点云分割预处理////////////////////////////////////////////////////////////////////////
vector<c_k_f> clouds_keys_features;

void com_key_feature(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float leaf_size = 1) {

	pcl::PointCloud<pcl::Normal>::Ptr normal(new pcl::PointCloud<pcl::Normal>());
	*normal = *normal_estimation_OMP(cloud, leaf_size*5.0f);
	pcl::PointCloud<pcl::PointXYZ>::Ptr key(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::Feature>::Ptr feature(new pcl::PointCloud<pcl::Feature>());
	//double start = 0;
	//double end = 0;
	//start = GetTickCount();
	*key = *key_detect(cloud, normal, 5.0f*leaf_size, 3.0f * leaf_size);
	*feature = *com_lsfh33_feature(cloud, normal, key, 10.0f * leaf_size);
	//end = GetTickCount();
	//cout << "源点云关键点数目：" << key_source->size() << endl;
	//cout << "源点云特征估计：" << end - start << "ms" << endl;
	c_k_f cloud_key_feature;
	cloud_key_feature.cloud = cloud;
	cloud_key_feature.key = key;
	cloud_key_feature.feature = feature;
	m.lock();
	clouds_keys_features.push_back(cloud_key_feature);
	m.unlock();
	return;
}

void com_k_f(vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clouds, float leaf_size = 1) {

	clouds_keys_features.clear();
	for (int i = 0; i < clouds.size(); i++) {
		thread t(com_key_feature, clouds[i], leaf_size);
		t.detach();
	}
	while (clouds_keys_features.size() != clouds.size()) {
		Sleep(10);
	}
	return;
}

/////////////////////filter/////////////////////////////
//int main() {
//	vector<string> names = { "armadillo", "bunny", "cat","centaur","cheff", "chicken","david","dog", "dragon","face",
//		"ganesha","gorilla","gun","horse","lioness","para" ,"trex","victoria","wolf" };
//	for (int i = 0; i < names.size(); i++) {
//		pcl::PointCloud<pcl::PointXYZ>::Ptr scene(new pcl::PointCloud<pcl::PointXYZ>);
//		pcl::io::loadPLYFile("D:/PCD/识别点云角度修正/scene/scene/" + names[i] + ".ply", *scene);
//		*scene = *filter_to_leaf_size(scene, 1.0f);
//		pcl::io::savePLYFile("D:/PCD/识别点云角度修正/scene/filter/" + names[i] + ".ply", *scene);
//
//		pcl::PointCloud<pcl::PointXYZ>::Ptr model(new pcl::PointCloud<pcl::PointXYZ>);
//		pcl::io::loadPLYFile("D:/PCD/识别点云角度修正/model/model/" + names[i] + ".ply", *model);
//		*model = *filter_to_leaf_size(model, 1.0f);
//		pcl::io::savePLYFile("D:/PCD/识别点云角度修正/model/filter/" + names[i] + ".ply", *model);
//	}
//	return 0;
//}


/////////////////////离线计算/////////////////////////////////////////////////////////////
//int main(int argc, char** argv) {
//	string road = "D:/code/PCD/识别点云/model/";
//	vector<string> names = { "armadillo", "bunny", "cat","centaur","cheff", "chicken","david","dog", "dragon","face",
//		"ganesha","gorilla","gun","horse","lioness","para" ,"trex","victoria","wolf" };
//
//	for (int i = 0; i < names.size(); i++) {
//		string name = "D:/PCD/识别点云角度修正/model/filter/" + names[i] + ".ply";
//		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
//		pcl::io::loadPLYFile(name, *cloud);
//		float leaf_size = 1;
//
//		pcl::PointCloud<pcl::Normal>::Ptr normal(new pcl::PointCloud<pcl::Normal>());
//		normal = normal_estimation_OMP(cloud, leaf_size*5.0f);
//		pcl::PointCloud<pcl::Feature>::Ptr feature(new pcl::PointCloud<pcl::Feature>());
//		pcl::PointCloud<pcl::PointXYZ>::Ptr key(new pcl::PointCloud<pcl::PointXYZ>());
//		pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
//		tree->setInputCloud(cloud);
//		*key = *key_detect(cloud, normal, 5.0f * leaf_size, 3.0f * leaf_size);
//		cout << names[i] << ": " << key->size() << endl;
//		*feature = *com_lsfh33_feature(cloud, normal, key, 10.0f * leaf_size);
//		pcl::io::savePLYFile("D:/PCD/识别点云角度修正/model/key/" + names[i] + "_key.ply", *key);
//		pcl::io::savePLYFile("D:/PCD/识别点云角度修正/model/feature/" + names[i] + "_feature.ply", *feature);
//	}
//	return 0;
//}

//随即旋转平移
pcl::PointCloud<pcl::PointXYZ>::Ptr random_transform(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
	float rx, ry, rz, tx, ty, tz;
	rx = rand() % 360;
	ry = rand() % 360;
	rz = rand() % 360;
	tx = rand() % 60 - 30;
	ty = rand() % 60 - 30;
	tz = rand() % 60 - 30;
	rx = rx / 180.0f*M_PI;
	ry = ry / 180.0f*M_PI;
	rz = rz / 180.0f*M_PI;
	Eigen::Affine3f RT = Eigen::Affine3f::Identity();
	RT.rotate(Eigen::AngleAxisf(rx, Eigen::Vector3f::UnitX()));
	RT.rotate(Eigen::AngleAxisf(ry, Eigen::Vector3f::UnitY()));
	RT.rotate(Eigen::AngleAxisf(rz, Eigen::Vector3f::UnitZ()));
	RT.translation() << tx, ty, tz;
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::transformPointCloud(*cloud, *cloud_out, RT);
	return cloud_out;
}

//////////////////////识别///////////////////////////////////////////////////////////////////////////////////////////
//int main(int argc, char** argv) {
//	vector<string> names = { "armadillo", "bunny", "cat","centaur","cheff", "chicken","david","dog", "dragon","face",
//	"ganesha","gorilla","horse","para" ,"trex","wolf" };
//
//	//	vector<string> names = { "armadillo", "bunny", "cat","centaur","cheff", "chicken","david","dog", "dragon","face",
//	//		"ganesha","gorilla","gun","horse","lioness","para" ,"trex","victoria","wolf" };
//
//	vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> models;
//	vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> keys;
//	vector<pcl::PointCloud<pcl::FPFHSignature33>::Ptr> features;
//	for (int i = 0; i < names.size(); i++) {
//		pcl::PointCloud<pcl::PointXYZ>::Ptr model(new pcl::PointCloud<pcl::PointXYZ>);
//		pcl::PointCloud<pcl::PointXYZ>::Ptr key(new pcl::PointCloud<pcl::PointXYZ>);
//		pcl::PointCloud<pcl::FPFHSignature33>::Ptr feature(new pcl::PointCloud<pcl::FPFHSignature33>);
//		pcl::io::loadPLYFile("D:/PCD/识别点云角度修正/model/filter/" + names[i] + ".ply", *model);
//		pcl::io::loadPLYFile("D:/PCD/识别点云角度修正/model/key/" + names[i] + "_key.ply", *key);
//		pcl::io::loadPLYFile("D:/PCD/识别点云角度修正/model/feature/" + names[i] + "_feature.ply", *feature);
//		models.push_back(model);
//		keys.push_back(key);
//		features.push_back(feature);
//	}
//	vector<vector<int>> res(names.size(), vector<int>(names.size(), 0));
//	int c = 0;
//	cout << "输入识别次数：";
//	cin >> c;
//
//	int noise = 0;
//	cout << "输入噪声：";
//	cin >> noise;
//
//	for (int k = 0; k < names.size();k++) {
//		string name;
//		name = names[k];
//		pcl::PointCloud<pcl::PointXYZ>::Ptr scenes(new pcl::PointCloud<pcl::PointXYZ>);
//		pcl::io::loadPLYFile("D:/PCD/识别点云角度修正/scene/filter/" + name + ".ply", *scenes);
//		int num = 0;
//		float t = 0;
//		for (int i = 0; i < c; i++) {
//			//show_point_cloud(scenes);
//			//show_point_cloud(scenes_temp);
//			float t1, t2, t3, t4;
//			t1 = GetTickCount();
//			vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clouds;
//			float leaf_size = 1.0f;
//			clouds.push_back(add_gaussian_noise(random_transform(scenes), noise));
//			////////////////特征提取/////////////////////////////////////////////////
//			t2 = GetTickCount();
//			vector<i_p_t> result_final;
//			com_k_f(clouds, leaf_size);
//			/////////////////场景识别//////////////////////////////////////////////////////////
//			t3 = GetTickCount();
//			for (int i = 0; i < clouds.size(); i++) {
//				result_final.push_back(predict(clouds_keys_features[i].cloud, clouds_keys_features[i].key,
//					clouds_keys_features[i].feature, names, models, keys, features));
//				if (result_final[i].percent > 0.9) {
//					cout << names[result_final[i].i] << endl;
//					res[result_final[i].i][k]++;
//					if (names[result_final[i].i] == name) {
//						num++;
//					}
//				}
//				else
//					cout << "nnnnnnnnnn" << endl;
//			}
//			t4 = GetTickCount();
//			cout << "随机变换时间：" << (t2 - t1) << "ms" << endl;
//			cout << "计算特征时间：" << (t3 - t2) << "ms" << endl;
//			cout << "识别时间：" << (t4 - t3) << "ms" << endl;
//			cout << "////////////////////////////////////////////////////////" << endl;
//			t += t4 - t1;
//			//show_point_clouds(clouds);
//			//show_point_clouds_and_trans_models(clouds, models, result_final);
//		}
//		cout << name << "的识别率：" << (float)num / (float)c*100.0f << "%" << endl;
//		cout << name << "平均识别时间：" << t / (float)c << "ms" << endl;
//		cout << "////////////////////////////////////////////////////////" << endl;
//	}
//	for (int i = 0; i < names.size(); i++) {
//		for (int j = 0; j < names.size(); j++) {
//			cout << res[i][j] << " ";
//		}
//		cout << endl;
//	}
//	return 0;
//}

//int main(int argc, char** argv) {
//	vector<string> names = { "armadillo", "bunny", "cat","centaur","cheff", "chicken","david","dog", "dragon","face",
//	"ganesha","gorilla","horse","para" ,"trex","wolf" };
//
//	//	vector<string> names = { "armadillo", "bunny", "cat","centaur","cheff", "chicken","david","dog", "dragon","face",
//	//		"ganesha","gorilla","gun","horse","lioness","para" ,"trex","victoria","wolf" };
//
//	vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> models;
//	vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> keys;
//	vector<pcl::PointCloud<pcl::FPFHSignature33>::Ptr> features;
//	for (int i = 0; i < names.size(); i++) {
//		pcl::PointCloud<pcl::PointXYZ>::Ptr model(new pcl::PointCloud<pcl::PointXYZ>);
//		pcl::PointCloud<pcl::PointXYZ>::Ptr key(new pcl::PointCloud<pcl::PointXYZ>);
//		pcl::PointCloud<pcl::FPFHSignature33>::Ptr feature(new pcl::PointCloud<pcl::FPFHSignature33>);
//		pcl::io::loadPLYFile("D:/PCD/识别点云角度修正/model/filter/" + names[i] + ".ply", *model);
//		pcl::io::loadPLYFile("D:/PCD/识别点云角度修正/model/key/" + names[i] + "_key.ply", *key);
//		pcl::io::loadPLYFile("D:/PCD/识别点云角度修正/model/feature/" + names[i] + "_feature.ply", *feature);
//		models.push_back(model);
//		keys.push_back(key);
//		features.push_back(feature);
//	}
//	string name;
//	cout << "输入场景点云：";
//	while (cin >> name) {
//		pcl::PointCloud<pcl::PointXYZ>::Ptr scenes(new pcl::PointCloud<pcl::PointXYZ>);
//		pcl::io::loadPLYFile("D:/PCD/识别点云角度修正/scene/filter/" + name + ".ply", *scenes);
//		int num = 0;
//		float t = 0;
//		int c = 0;
//		cout << "输入识别次数：";
//		cin >> c;
//		for (int i = 0; i < c; i++) {
//			pcl::PointCloud<pcl::PointXYZ>::Ptr scenes_temp(new pcl::PointCloud<pcl::PointXYZ>);
//			scenes_temp = random_transform(scenes);
//			//show_point_cloud(scenes);
//			//show_point_cloud(scenes_temp);
//			float t1, t2, t3, t4;
//			///////////////欧式聚类////////////////////////////////////////
//			t1 = GetTickCount();
//			vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clouds;
//			float leaf_size = 1.0f;
//			clouds = euclidean_cluster(scenes_temp, 10.0f*leaf_size);
//			////////////////特征提取/////////////////////////////////////////////////
//			t2 = GetTickCount();
//			vector<i_p_t> result_final;
//			com_k_f(clouds, leaf_size);
//			/////////////////场景识别//////////////////////////////////////////////////////////
//			t3 = GetTickCount();
//			for (int i = 0; i < clouds.size(); i++) {
//				result_final.push_back(predict(clouds_keys_features[i].cloud, clouds_keys_features[i].key,
//					clouds_keys_features[i].feature, names, models, keys, features));
//				if (result_final[i].percent > 0.9) {
//					cout << names[result_final[i].i] << endl;
//					if (names[result_final[i].i] == name) {
//						num++;
//					}
//				}
//				else
//					cout << "nnnnnnnnnn" << endl;
//			}
//			t4 = GetTickCount();
//			cout << "欧式聚类时间：" << (t2 - t1) << "ms" << endl;
//			cout << "计算特征时间：" << (t3 - t2) << "ms" << endl;
//			cout << "识别时间：" << (t4 - t3) << "ms" << endl;
//			cout << "////////////////////////////////////////////////////////" << endl;
//			t += t4 - t1;
//			//show_point_clouds(clouds);
//			//show_point_clouds_and_trans_models(clouds, models, result_final);
//		}
//		cout << name << "的识别率：" << (float)num / (float)c*100.0f << "%" << endl;
//		cout << name << "平均识别时间：" << t / (float)c << "ms" << endl;
//		cout << "////////////////////////////////////////////////////////" << endl;
//	}
//
//	return 0;
//}

///////////////////配准
int main() {
	string model_name, scene_name;
	float leaf_size = 1.0f;
	float noise = 0;
	while (cin >>  model_name>> noise) {
		//pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_scene(new pcl::PointCloud<pcl::PointXYZ>);
		//pcl::io::loadPLYFile("D:/PCD/识别点云角度修正/scene/filter/" + scene_name + ".ply", *cloud_scene);
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_model(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::io::loadPLYFile("D:/PCD/识别点云角度修正/model/filter/" + model_name + ".ply", *cloud_model);
		*cloud_model = *add_gaussian_noise(cloud_model, noise);

		pcl::PointCloud<pcl::Normal>::Ptr normal_model(new pcl::PointCloud<pcl::Normal>);
		*normal_model = *normal_estimation_OMP(cloud_model, 5.0f*leaf_size);
		pcl::PointCloud<pcl::PointXYZ>::Ptr key_model(new pcl::PointCloud<pcl::PointXYZ>);
		*key_model = *key_detect(cloud_model, normal_model, 5.0f*leaf_size, 5.0f * leaf_size);
		cout << "模型关键点：" << key_model->size() << endl;
		pcl::PointCloud<pcl::FPFHSignature33>::Ptr feature_model(new pcl::PointCloud<pcl::FPFHSignature33>);
		*feature_model = *com_lsfh33_feature(cloud_model, normal_model, key_model, 10.0f*leaf_size);

		//pcl::PointCloud<pcl::Normal>::Ptr normal_scene(new pcl::PointCloud<pcl::Normal>);
		//*normal_scene = *normal_estimation_OMP(cloud_scene, 5.0f*leaf_size);
		//pcl::PointCloud<pcl::PointXYZ>::Ptr key_scene(new pcl::PointCloud<pcl::PointXYZ>);
		//*key_scene = *key_detect(cloud_scene, normal_scene, 5.0f*leaf_size, 5.0f * leaf_size);
		//cout << "场景关键点：" << key_scene->size() << endl;
		//pcl::PointCloud<pcl::FPFHSignature33>::Ptr feature_scene(new pcl::PointCloud<pcl::FPFHSignature33>);
		//*feature_scene = *com_lsfh33_feature(cloud_scene,  normal_scene,key_scene, 10.0f*leaf_size);
		//show_key_scene(cloud_scene, key_scene);
		show_key_model(cloud_model, key_model);
		//my_align(cloud_scene, key_scene, feature_scene, cloud_model, key_model, feature_model, leaf_size);

	}
	return 0;
}