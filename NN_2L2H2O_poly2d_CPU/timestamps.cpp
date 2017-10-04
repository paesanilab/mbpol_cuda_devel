#include "timestamps.h"
#include <chrono>
#include <string>
#include <iostream>
#include <time.h>
#include <climits>
#include <iomanip>

using namespace std;

timestamp::timestamp(int _markid, std::string _label): markid(_markid),label(_label){
     timespan=0;
};
timestamp::~timestamp(){};
timestamp::timestamp(const timestamp& that){
    start=that.start;
    end=that.end;
    timespan = that.timespan;
    markid = that.markid;
    label = that.label;
}



void timestamp::setthread(int _id){
     markid=_id;
}


void timestamp::setlabel(string _label){
     label = _label;
};

void timestamp::stampstart(){
     start = chrono::high_resolution_clock::now();
};

void timestamp::stampend(){
     end = chrono::high_resolution_clock::now();  
     timespan = (long long int)chrono::duration_cast<chrono::microseconds>(end-start).count();
};
     


timers_t::timers_t(){
     //srand(time(NULL));
     timers_list.clear();
     timecollections.clear();
     internal_random_id=1000;  // 0~999 : reserved id. Internal random ID starts from 1000. 
}

timers_t::~timers_t(){
     timers_list.clear();
     timecollections.clear();
}

timers_t::timers_t(const timers_t& that){
     timers_list.clear();
     for(auto item : that.timers_list){
          timers_list.insert(item);
     }
     timecollections.clear();
     for(auto item : that.timecollections){
          timecollections.insert(item);
     }
     internal_random_id=that.internal_random_id;
}


// insert a timer into the list with a given unique ID
bool timers_t::insert_timer(timerid_t _id, int _markid, string _label){
     if (timers_list.find(_id)==timers_list.end()) {          
          timestamp stamp(_markid,_label);                             
          timers_list.insert({_id, stamp});          
          return true;
     }
     return false;
}  

// insert a timer into the list with a random ID, and return with this id.
void timers_t::insert_random_timer(timerid_t & id, int _markid, std::string _label){
     for(bool ifinsert=false; !(ifinsert); ) {
          internal_random_id++;
          ifinsert = insert_timer(internal_random_id, _markid, _label);     
     }
     id = internal_random_id;
     return;
}


bool timers_t::timer_start(timerid_t _id){
     if (timers_list.find(_id)==timers_list.end()) {
          return false;
     }
     timers_list[_id].stampstart();
     return true;
}


bool timers_t::timer_end(timerid_t _id, bool ifadd, bool ifsave){
     if (timers_list.find(_id)==timers_list.end()) {
          return false;
     }
     timers_list[_id].stampend();
     
     if(ifadd){     
          add_time(_id);
     }     
     if(!ifsave){
          timers_list.erase(_id);
     }     
     return true;          
}


long long int timers_t::get_time_span(timerid_t _id){
     if (timers_list.find(_id)==timers_list.end()) {
          return 0;
     }
     return timers_list[_id].timespan;
}

timerid_t timers_t::get_thread_id(timerid_t _id){
     if (timers_list.find(_id)==timers_list.end()) {
          return 0;
     }
     return timers_list[_id].markid;
}

string timers_t::get_label(timerid_t _id){
     if (timers_list.find(_id)==timers_list.end()) {
          return 0;
     }
     return timers_list[_id].label;
}

bool timers_t::get_all_timers_info(){     
     for (auto item : timers_list) {
          cout << setw(4)  << item.second.markid   
               << setw(10) << item.second.label
               << setw(16) << item.second.timespan
               << endl;
     };    
     return true;
}


int timers_t::get_num_timers(){
     return timers_list.size();
}


void timers_t::get_all_timers_id(){
     for (auto item : timers_list) {
          cout << setw(20) << item.first << endl;
     };
     return;
};

void timers_t::get_time_collections(){
     cout << " All timers information : " << endl;
     for(auto itr = timecollections.begin(); itr != timecollections.end(); itr++) {
          cout << "  TimerMark= " << setw(3) << itr->markid 
               << "    Label= " << left << setw(25) << itr->label
               << "    Time[MuS]= " << setw(15) << itr->timespan
               << endl;
     }
};

bool timers_t::add_time(timerid_t _id){
     timestamp& target = timers_list[_id];
     auto findtarget = timecollections.find(target);
     if ( findtarget == timecollections.end()){
          //If not find, initialize a new record in timecollections
          timecollections.insert(target);
     }else {
          //If find, add time to timecollections
          findtarget->timespan += target.timespan;
     }
     return true;
};


