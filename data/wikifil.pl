#!/usr/bin/perl
use strict;
use warnings;

# 判断是否在 <text> 标签内部
my $in_text = 0;
my $buffer = '';

while (<>) {
    # 检测进入 <text> 标签（去掉 ^ 符号，允许匹配行中任意位置）
    if (/<text/) {
        $in_text = 1;
        $buffer = '';
        next;  # 跳过 <text> 开始标签
    }

    # 检测退出 </text> 标签（去掉 ^ 符号，允许匹配行中任意位置）
    if (/<\/text>/) {
        $in_text = 0;

        # 处理并输出 buffer 中的内容
        if ($buffer =~ /\S/) {  # 只有非空才输出
            # 替换转义字符
            $buffer =~ s/\&\#39;/\'/g;
            $buffer =~ s/\&quot;/\"/g;
            $buffer =~ s/\&lt;/</g;
            $buffer =~ s/\&gt;/>/g;
            $buffer =~ s/\&amp;/\&/g;

            # 删除所有 HTML 标签
            $buffer =~ s/<[^>]+>//g;

            # 去掉前后空白
            $buffer =~ s/^\s+//;
            $buffer =~ s/\s+$//;

            # 将多个空格或换行替换为单个空格
            $buffer =~ s/\s+/ /g;

            print "$buffer\n";
        }
        next;
    }

    # 如果在 <text> 内部，则累加内容
    if ($in_text) {
        $buffer .= $_;
    }
}