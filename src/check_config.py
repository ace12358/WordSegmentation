#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
check_config.py
設定ファイル（ini file）をチェックするユーティリティ

注意！
エラーチェックはいい加減です。
'''

import configparser
import os
import sys


def show_config(ini):
    '''
    設定ファイルの全ての内容を表示する（コメントを除く）
    '''
    for section in ini.sections():
        print ('[%s]' % (section))
        show_sectoin(ini, section)
    return


def show_sectoin(ini, section):
    '''
    設定ファイルの特定のセクションの内容を表示する
    '''
    for key in ini.options(section):
        show_key(ini, section, key)
    return


def show_key(ini, section, key):
    '''
    設定ファイルの特定セクションの特定のキー項目（プロパティ）の内容を表示する
    '''
    print ('%s.%s =%s' % (section, key, ini.get(section, key)))
    return


def set_value(ini, section, key, value):
    '''
    設定ファイルの特定セクションの特定のキー項目（プロパティ）の内容を変更する
    '''
    ini.set(section, key, value)
    print ('set %s.%s =%s' % (section, key, ini.get(section, key)))
    return


def usage():
    sys.stderr.write("Usage: %s inifile [section [key [value]]]\n" % sys.argv[0])
    return


if __name__ == '__main__':
    argc = len(sys.argv)
    if argc == 1:
        usage()
        sys.exit(1)

    # 設定ファイル読み込み
    INI_FILE = sys.argv[1]
    ini = configparser.SafeConfigParser()
    if os.path.exists(INI_FILE):
        ini.read(INI_FILE)
    else:
        sys.stderr.write('%s が見つかりません' % INI_FILE)
        sys.exit(2)

    if argc == 2:
        show_config(ini)
    elif argc == 3:
        show_sectoin(ini, sys.argv[2])
    elif argc == 4:
        show_key(ini, sys.argv[2], sys.argv[3])
    elif argc == 5:
        set_value(ini, sys.argv[2], sys.argv[3], sys.argv[4])
        # ファイルに書き出す（注意！現状だとコメントや改行を消してしまいます）
        f = open(INI_FILE, "w")
        ini.write(f)
        f.close()
    else:
        usage()
        sys.exit(3)

    sys.exit(0)
#EOF
