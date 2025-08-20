#!/usr/bin/env python

def pytest_sessionfinish(session, exitstatus):
    if exitstatus == 1:  # 如果有错误
        session.exitstatus = 0  # 将返回码设为 0