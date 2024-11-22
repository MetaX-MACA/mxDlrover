#include <curl/curl.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdlib>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "xpu_timer/common/manager.h"
#include "xpu_timer/common/xpu_timer.h"
#include "xpu_timer/common/util.h"

namespace atorch
{
  namespace
  {

    using namespace testing;

    class TimerStub: public XpuTimer
    {
      public:
      TimerStub() = default;
      ~TimerStub() = default;
      uint64_t getDuration() { return duration;}
      bool isReady() {return true;};
      const std::string getName() { return name;};
      const std::string getType() { return "";};
      const std::string getFlop() { return "";};
      void startRecord() {return;};
      void endRecord() {return;};

      uint64_t duration = 0;
      std::string name{"stub"};
    };


    class IntegrationTest : public testing::Test
    {
    public:
      void SetUp() override
      {
        setenv("XPU_TIMER_OPENBRPC", "1", 1);
      }

      void TearDown() override
      {
      }
      struct Response
      {
        long code = 0;
        std::string body;
        std::string contentType;
      };

      std::function<void(CURL *)> fetchPrePerform_;

      Response FetchMetrics(const std::string &metrics_path) const
      {
        auto curl = std::shared_ptr<CURL>(curl_easy_init(), curl_easy_cleanup);
        if (!curl)
        {
          throw std::runtime_error("failed to initialize libcurl");
        }

        const auto url = base_url_ + metrics_path;
        Response response;

        curl_easy_setopt(curl.get(), CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl.get(), CURLOPT_WRITEDATA, &response.body);
        curl_easy_setopt(curl.get(), CURLOPT_WRITEFUNCTION, WriteCallback);

        if (fetchPrePerform_)
        {
          fetchPrePerform_(curl.get());
        }

        CURLcode curl_error = curl_easy_perform(curl.get());
        if (curl_error != CURLE_OK)
        {
          throw std::runtime_error("failed to perform HTTP request");
        }

        curl_easy_getinfo(curl.get(), CURLINFO_RESPONSE_CODE, &response.code);

        char *ct = nullptr;
        curl_easy_getinfo(curl.get(), CURLINFO_CONTENT_TYPE, &ct);
        if (ct)
        {
          response.contentType = ct;
        }

        return response;
      }

      std::string base_url_ = "http://127.0.0.1";
      std::string default_metrics_path_ = ":28888/vars";

    private:
      static std::size_t WriteCallback(void *contents, std::size_t size,
                                       std::size_t nmemb, void *userp)
      {
        auto response = reinterpret_cast<std::string *>(userp);

        std::size_t realsize = size * nmemb;
        response->append(reinterpret_cast<const char *>(contents), realsize);
        return realsize;
      }
    };


    TEST_F(IntegrationTest, recordEventNormal)
    {
      auto event = GpuTimerManager<TimerStub>::getInstance().getEvent();
      LOG(INFO) << event;
      event->duration = 10;
      event->name = "ut_for_one";


      auto event2 = GpuTimerManager<TimerStub>::getInstance().getEvent();
      LOG(INFO) << event2;
      event2->duration = 20;
      event2->name = "ut_for_one";
    

      auto event3 = GpuTimerManager<TimerStub>::getInstance().getEvent();
      LOG(INFO) << event2;
      event3->duration = 20;
      event3->name = "ut_for_one";

      GpuTimerManager<TimerStub>::getInstance().recordEvent(event);
      sleep(1);
      GpuTimerManager<TimerStub>::getInstance().recordEvent(event2);
      sleep(1);
      GpuTimerManager<TimerStub>::getInstance().recordEvent(event3); 
      while (1)
      {
        const auto metrics = FetchMetrics(default_metrics_path_);

        if (metrics.code!=200)
          continue;
        if (!metrics.body.empty())
        {
          if (metrics.body.find("ut_for_one_count : 3") != std::string::npos)
          {
            if (metrics.body.find("ut_for_one_latency : 0") != std::string::npos)
            {
              sleep(0.5);
              continue;
            }
            EXPECT_THAT(metrics.body, HasSubstr("ut_for_one_latency"));

            // LOG(INFO) << metrics.body;
            sleep(0.5);
            auto prom_metrics = FetchMetrics(":38888/metrics");
            // LOG(INFO) << prom_metrics.body;
            EXPECT_THAT(prom_metrics.body, HasSubstr("ut_for_one_avg_latency"));
            break;
          }
        }
        sleep(0.1);
      }
    }
  } // namespace
} // namespace prometheus
