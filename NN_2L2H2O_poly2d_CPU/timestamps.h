#ifndef TIMESTAMPS_H
#define TIMESTAMPS_H


#include <chrono>
#include <map>
#include <set>
#include <string>


// Time stamp class. Recording the start/end/passed time, the label, used by which thread of a timer
struct timestamp {
     std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
     std::string label;
     mutable long long int timespan;  // In [MuS] unit
     int markid;
     
     timestamp(int id=0, std::string label="");
     ~timestamp();
     timestamp(const timestamp& that);
     //timestamp operator=(const timestamp& that);
     
     void setthread(int id);
     void setlabel(std::string label);
     void stampstart();
     void stampend();
};

// Helper function to determine the sortation of the time stamp by the thread id and the label.
// Used to identify timers in set container.
struct compare_by_label {
     bool operator () (const timestamp & lhs, const timestamp & rhs){
          if (lhs.markid < rhs.markid) {
               return true;
          } else if ((lhs.markid == rhs.markid) && (lhs.label < rhs.label)){
               return true;
          }
          return false;
     };
};


typedef unsigned long long int timerid_t;


// Timer collection class, ready for use by external calls. 
class timers_t {
private:
     std::map<timerid_t, timestamp> timers_list;   // A map container saving all timers that can be identified by a unique unsigned long long int id.
     std::set<timestamp, compare_by_label> timecollections;      // Collection container accumulating the time span in all timers. 
                                                                 // Use thread and label to distinguish each other.
                                                                 
     timerid_t internal_random_id;                                                                 
                                                                 
public:     
     
     timers_t();
     ~timers_t();
     timers_t(const timers_t& that);

// Following functions are used to deal with a timer according to its unique id:     
     bool insert_timer(timerid_t _id, int markid=0, std::string _label="");  // Insert a timer by a specific id
     void insert_random_timer(timerid_t & _id, int markid=0, std::string _label=""); // Insert a timer with a random id, which is returned as reference.
     bool timer_start(timerid_t _id);  // Start a timer.
     bool timer_end(timerid_t _id, bool ifadd=true, bool ifsave=false);      // End a timer. 
                                                                      // IFADD  : the recorded time span will be added to the timecollection
                                                                      // IFSAVE : the timer will be saved in the timers_list (default: the timer will be removed)
// Get timer information based on the id:
     long long int get_time_span(timerid_t _id);
     timerid_t get_thread_id(timerid_t _id);
     std::string get_label(timerid_t _id);
     

// Helper function to get timers info:     
     int get_num_timers();
     bool get_all_timers_info();
     void get_all_timers_id();
     void get_time_collections();
     
// Add time to collection according to thread and label
     bool add_time(timerid_t _id);     
          
};

#endif


