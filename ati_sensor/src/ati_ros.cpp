#include <iostream>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <ros/ros.h>
#include <geometry_msgs/WrenchStamped.h>
#include <time.h>


#include "jsd/jsd_pub.h"
#include "jsd/jsd_ati_fts_pub.h"

using namespace std;

#define takeoff_alt 1.6
#define motion_dist 1.5

jsd_t* jsd;

int main(int argc, char** argv)
{
	ros::init(argc, argv, "ati_pub");
	ros::NodeHandle nh("~");
	
	//Initialization - Rate
	int RATE = 1000;
	nh.getParam("trajectory_rate", RATE);
	ros::Rate loop_rate(RATE);
	
	//Initialization - Sensor name
	std::string object_name = "ati_mini40";
	nh.getParam("object_name",object_name);
	
	//Initialization - Interface name
	std::string ifname = "eno1";
	nh.getParam("ifname",ifname);
	
	//Initialization - slave_id
	int slave_id = 1;
	nh.getParam("slave_id", slave_id);
	
	// set device configuration here
	jsd_slave_config_t my_config = {0};
	my_config.ati_fts.calibration = 0;
	snprintf(my_config.name, JSD_NAME_LEN, "%s", object_name.c_str());
	my_config.configuration_active = true;
	my_config.product_code         = JSD_ATI_FTS_PRODUCT_CODE;

	// Initialize jsd element
	jsd = jsd_alloc();
	jsd_set_slave_config(jsd, slave_id, my_config);
	if (!jsd_init(jsd, ifname.c_str(), 1)) {ROS_ERROR("Could not init jsd");return -1;}
	
	// Initialization - Publishers/Subscribers
	ros::Publisher ft_meas_ = nh.advertise<geometry_msgs::WrenchStamped>(object_name + "/ft_meas", 1);
	geometry_msgs::WrenchStamped ft_meas;
	ft_meas.header.frame_id = object_name;
	ft_meas.header.stamp = ros::Time::now();
	
	ROS_INFO("Starting MAIN LOOP");
	uint32_t jsd_iter = 0;
	
	while(ros::ok()){
	      jsd_read(jsd, EC_TIMEOUTRET);
	      jsd_ati_fts_read(jsd, slave_id);
	      const jsd_ati_fts_state_t* state = jsd_ati_fts_get_state(jsd, slave_id);
	      
	      ROS_INFO("(%lf, %lf, %lf)  (%lf, %lf, %lf)  fault= %u counter=%u", state->fx, state->fy, state->fz, state->tx, state->ty, state->tz, state->active_error, state->sample_counter);

	      jsd_write(jsd);
	      jsd_iter++;
	      
	      //Format pub message
	      ft_meas.header.seq = jsd_iter;
	      ft_meas.header.stamp = ros::Time::now();
	      
	      ft_meas.wrench.force.x = state->fx;
	      ft_meas.wrench.force.y = state->fy;
	      ft_meas.wrench.force.z = state->fz;
	      ft_meas.wrench.torque.x = state->tx;
	      ft_meas.wrench.torque.y = state->ty;
	      ft_meas.wrench.torque.z = state->tz;
	      
	      //Publish
	      ft_meas_.publish(ft_meas);
	      ros::spinOnce();
	      loop_rate.sleep();
	  
	}
	return 0;
}