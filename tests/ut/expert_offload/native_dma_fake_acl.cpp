// Test-only CANN substitute; loaded only by the isolated CPU subprocess.
#include <atomic>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <vector>
static std::mutex mutex;
static std::vector<uintptr_t> order;
extern "C" {
int aclrtGetCurrentContext(void** p) {*p=reinterpret_cast<void*>(1);return 0;}
int aclrtSetCurrentContext(void*) {return 0;}
int aclrtMemcpyAsync(void* dst,size_t maximum,const void* src,size_t n,int kind,void*) {
 if(kind!=1 || n>maximum)return 1;
 std::lock_guard<std::mutex> lock(mutex);std::memcpy(dst,src,n);order.push_back(reinterpret_cast<uintptr_t>(dst));return 0;
}
int aclrtValueWrite(void* ptr,uint64_t v,uint32_t,void*) {
 std::lock_guard<std::mutex> lock(mutex);order.push_back(reinterpret_cast<uintptr_t>(ptr));
 __atomic_store_n(static_cast<uint64_t*>(ptr),v,__ATOMIC_RELEASE);return 0;
}
int aclrtSynchronizeStream(void*) {return 0;}
size_t fake_order_size(){std::lock_guard<std::mutex> lock(mutex);return order.size();}
uintptr_t fake_order_at(size_t i){std::lock_guard<std::mutex> lock(mutex);return order.at(i);}
}
