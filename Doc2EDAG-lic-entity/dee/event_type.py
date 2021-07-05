# -*- coding: utf-8 -*-
# AUTHOR: Shun Zheng
# DATE: 19-9-19


class BaseEvent(object):
    def __init__(self, fields, event_name='Event', key_fields=(), recguid=None):
        self.recguid = recguid
        self.name = event_name
        self.fields = list(fields)
        self.field2content = {f: None for f in fields}
        self.nonempty_count = 0
        self.nonempty_ratio = self.nonempty_count / len(self.fields)

        self.key_fields = set(key_fields)
        for key_field in self.key_fields:
            assert key_field in self.field2content

    def __repr__(self):
        event_str = "\n{}[\n".format(self.name)
        event_str += "  {}={}\n".format("recguid", self.recguid)
        event_str += "  {}={}\n".format("nonempty_count", self.nonempty_count)
        event_str += "  {}={:.3f}\n".format("nonempty_ratio", self.nonempty_ratio)
        event_str += "] (\n"
        for field in self.fields:
            if field in self.key_fields:
                key_str = " (key)"
            else:
                key_str = ""
            event_str += "  " + field + "=" + str(self.field2content[field]) + ", {}\n".format(key_str)
        event_str += ")\n"
        return event_str

    def update_by_dict(self, field2text, recguid=None):
        self.nonempty_count = 0
        self.recguid = recguid

        for field in self.fields:
            if field in field2text and field2text[field] is not None:
                self.nonempty_count += 1
                self.field2content[field] = field2text[field]
            else:
                self.field2content[field] = None

        self.nonempty_ratio = self.nonempty_count / len(self.fields)

    def field_to_dict(self):
        return dict(self.field2content)

    def set_key_fields(self, key_fields):
        self.key_fields = set(key_fields)

    def is_key_complete(self):
        for key_field in self.key_fields:
            if self.field2content[key_field] is None:
                return False

        return True

    def is_good_candidate(self):
        raise NotImplementedError()

    def get_argument_tuple(self):
        args_tuple = tuple(self.field2content[field] for field in self.fields)
        return args_tuple


class Event1(BaseEvent):
    NAME = '解除质押'
    FIELDS = [
        '质押方',
        '质押物',
        '质押股票/股份数量',
        '质押物所属公司',
        '质押物占总股比',
        '质押物占持股比',
        '质权方',
        '事件时间',
        '披露时间',
    ]

    def __init__(self, recguid=None):
        super().__init__(
            Event1.FIELDS, event_name=Event1.NAME, recguid=recguid
        )
        self.set_key_fields([
            '质押方',
            '质押物',
            #'质押股票/股份数量',
        ])

    def is_good_candidate(self, min_match_count=5):
        key_flag = self.is_key_complete()
        if key_flag:
            if self.nonempty_count >= min_match_count:
                return True
        return False


class Event2(BaseEvent):
    NAME = '股份回购'
    FIELDS = [
        '回购方',
        '回购股份数量',
        '每股交易价格',
        '交易金额',
        '回购完成时间'
        '占公司总股本比例',
        '披露时间',
    ]

    def __init__(self, recguid=None):
        super().__init__(
            Event2.FIELDS, event_name=Event2.NAME, recguid=recguid
        )
        self.set_key_fields([
            '回购方',
        ])

    def is_good_candidate(self, min_match_count=4):
        key_flag = self.is_key_complete()
        if key_flag:
            if self.nonempty_count >= min_match_count:
                return True
        return False


class Event3(BaseEvent):
    NAME = '股东减持'
    FIELDS = [
        '减持方',
        '股票简称',
        '交易股票/股份数量',
        '每股交易价格',
        '交易金额',
        '减持部分占所持比例',
        '减持部分占总股本比例',
        '交易完成时间',
        '披露时间'
    ]

    def __init__(self, recguid=None):
        super().__init__(
            Event3.FIELDS, event_name=Event3.NAME, recguid=recguid
        )
        self.set_key_fields([
            '减持方',
            '交易股票/股份数量',
        ])

    def is_good_candidate(self, min_match_count=4):
        key_flag = self.is_key_complete()
        if key_flag:
            if self.nonempty_count >= min_match_count:
                return True
        return False


class Event4(BaseEvent):
    NAME = '亏损'
    FIELDS = [
        '公司名称',
        '净亏损',
        '财报周期',
        '亏损变化',
        '披露时间',

    ]

    def __init__(self, recguid=None):
        super().__init__(
            Event4.FIELDS, event_name=Event4.NAME, recguid=recguid
        )
        self.set_key_fields([
            '公司名称',
            #'TradedShares',
        ])

    def is_good_candidate(self, min_match_count=4):
        key_flag = self.is_key_complete()
        if key_flag:
            if self.nonempty_count >= min_match_count:
                return True
        return False


class Event5(BaseEvent):
    NAME = '中标'
    FIELDS = [
        '中标公司',
        '中标标的',
        '招标方',
        '中标金额',
        '中标日期',
        '披露日期',

    ]

    def __init__(self, recguid=None):
        # super(EquityPledgeEvent, self).__init__(
        super().__init__(
            Event5.FIELDS, event_name=Event5.NAME, recguid=recguid
        )
        self.set_key_fields([
            '中标公司',
            '中标标的',

        ])

    def is_good_candidate(self, min_match_count=5):
        key_flag = self.is_key_complete()
        if key_flag:
            if self.nonempty_count >= min_match_count:
                return True
        return False

class Event6(BaseEvent):
    NAME = '高管变动'
    FIELDS = [
        '高管姓名',
        '高管职位',
        '任职公司',
        '变动类型',
        '变动后职位',
        '变动后公司名称',
        '事件时间',
        '披露日期',

    ]

    def __init__(self, recguid=None):
        # super(EquityPledgeEvent, self).__init__(
        super().__init__(
            Event6.FIELDS, event_name=Event6.NAME, recguid=recguid
        )
        self.set_key_fields([
            '高管姓名',
            '变动类型',

        ])

    def is_good_candidate(self, min_match_count=5):
        key_flag = self.is_key_complete()
        if key_flag:
            if self.nonempty_count >= min_match_count:
                return True
        return False

class Event7(BaseEvent):
    NAME = '企业破产'
    FIELDS = [
        '破产公司',
        '债权人',
        '债务规模',
        '破产时间',
        '披露时间',

    ]

    def __init__(self, recguid=None):
        # super(EquityPledgeEvent, self).__init__(
        super().__init__(
            Event7.FIELDS, event_name=Event7.NAME, recguid=recguid
        )
        self.set_key_fields([
            '破产公司',
        ])

    def is_good_candidate(self, min_match_count=5):
        key_flag = self.is_key_complete()
        if key_flag:
            if self.nonempty_count >= min_match_count:
                return True
        return False

class Event8(BaseEvent):
    NAME = '股东增持'
    FIELDS = [
        '增持方',
        '股票简称',
        '交易股票/股份数量',
        '每股交易价格',
        '交易金额',
        '增持部分占所持比例',
        '增持部分占总股本比例',
        '交易完成时间',
        '披露时间'
    ]

    def __init__(self, recguid=None):
        # super(EquityPledgeEvent, self).__init__(
        super().__init__(
            Event8.FIELDS, event_name=Event8.NAME, recguid=recguid
        )
        self.set_key_fields([
            '增持方',
            '交易股票/股份数量',
        ])

    def is_good_candidate(self, min_match_count=5):
        key_flag = self.is_key_complete()
        if key_flag:
            if self.nonempty_count >= min_match_count:
                return True
        return False

class Event9(BaseEvent):
    NAME = '被约谈'
    FIELDS = [
        '公司名称',
        '约谈机构',
        '被约谈时间',
        '披露时间',

    ]

    def __init__(self, recguid=None):
        # super(EquityPledgeEvent, self).__init__(
        super().__init__(
            Event9.FIELDS, event_name=Event9.NAME, recguid=recguid
        )
        self.set_key_fields([
            '公司名称',
            '约谈机构',
        ])

    def is_good_candidate(self, min_match_count=5):
        key_flag = self.is_key_complete()
        if key_flag:
            if self.nonempty_count >= min_match_count:
                return True
        return False

class Event10(BaseEvent):
    NAME = '企业收购'
    FIELDS = [
        '收购方',
        '被收购方',
        '收购标的',
        '交易金额',
        '收购完成时间',
        '披露时间',
    ]

    def __init__(self, recguid=None):
        # super(EquityPledgeEvent, self).__init__(
        super().__init__(
            Event10.FIELDS, event_name=Event10.NAME, recguid=recguid
        )
        self.set_key_fields([
            '收购方',
            '被收购方',
        ])

    def is_good_candidate(self, min_match_count=5):
        key_flag = self.is_key_complete()
        if key_flag:
            if self.nonempty_count >= min_match_count:
                return True
        return False

class Event11(BaseEvent):
    NAME = '公司上市'
    FIELDS = [
        '上市公司',
        '证券代码',
        '发行价格',
        '市值',
        '募资金额',
        '事件时间',
        '披露时间',
    ]

    def __init__(self, recguid=None):
        # super(EquityPledgeEvent, self).__init__(
        super().__init__(
            Event11.FIELDS, event_name=Event11.NAME, recguid=recguid
        )
        self.set_key_fields([
            '上市公司',
        ])

    def is_good_candidate(self, min_match_count=5):
        key_flag = self.is_key_complete()
        if key_flag:
            if self.nonempty_count >= min_match_count:
                return True
        return False

class Event12(BaseEvent):
    NAME = '企业融资'
    FIELDS = [
        '被投资方',
        '投资方',
        '领投方',
        '融资金额',
        '融资轮次',
        '事件时间',
        '披露时间',
    ]

    def __init__(self, recguid=None):
        # super(EquityPledgeEvent, self).__init__(
        super().__init__(
            Event12.FIELDS, event_name=Event12.NAME, recguid=recguid
        )
        self.set_key_fields([
            '融资金额',

        ])

    def is_good_candidate(self, min_match_count=5):
        key_flag = self.is_key_complete()
        if key_flag:
            if self.nonempty_count >= min_match_count:
                return True
        return False

class Event13(BaseEvent):
    NAME = '质押'
    FIELDS = [
        '质押方',
        '质押物',
        '质押股票/股份数量',
        '质押物所属公司',
        '质押物占总股比',
        '质押物占持股比',
        '质权方',
        '事件时间',
        '披露时间',
    ]

    def __init__(self, recguid=None):
        # super(EquityPledgeEvent, self).__init__(
        super().__init__(
            Event13.FIELDS, event_name=Event13.NAME, recguid=recguid
        )
        self.set_key_fields([
            '质押方',
            '质押物',
        ])

    def is_good_candidate(self, min_match_count=5):
        key_flag = self.is_key_complete()
        if key_flag:
            if self.nonempty_count >= min_match_count:
                return True
        return False

common_fields = []


event_type2event_class = {
    Event1.NAME: Event1,
    Event2.NAME: Event2,
    Event3.NAME: Event3,
    Event4.NAME: Event4,
    Event5.NAME: Event5,
    Event6.NAME: Event6,
    Event7.NAME: Event7,
    Event8.NAME: Event8,
    Event9.NAME: Event9,
    Event10.NAME: Event10,
    Event11.NAME: Event11,
    Event12.NAME: Event12,
    Event13.NAME: Event13,
}


event_type_fields_list = [
    (Event1.NAME, Event1.FIELDS),
    (Event2.NAME, Event2.FIELDS),
    (Event3.NAME, Event3.FIELDS),
    (Event4.NAME, Event4.FIELDS),
    (Event5.NAME, Event5.FIELDS),
    (Event6.NAME, Event6.FIELDS),
    (Event7.NAME, Event7.FIELDS),
    (Event8.NAME, Event8.FIELDS),
    (Event9.NAME, Event9.FIELDS),
    (Event10.NAME, Event10.FIELDS),
    (Event11.NAME, Event11.FIELDS),
    (Event12.NAME, Event12.FIELDS),
    (Event13.NAME, Event13.FIELDS),
]

field2biolabel = {'人名':["高管姓名"],\
                '公司名':['质权方',"质押物所属公司", "回购方", "股票简称", "公司名称", "中标公司", "招标方", "任职公司", "变动后公司名称", "破产公司", "约谈机构", "收购方", "被收购方", "上市公司","被投资方" ,"领投方"],\
                '数字':["质押股票/股份数量", "质押物占总股比", "质押物占持股比", "回购股份数量", "占公司总股本比例", "交易股票/股份数量", "减持部分占所持比例", "减持部分占总股本比例", "增持部分占所持比例", "增持部分占总股本比例", "证券代码"],\
                '金钱':["每股交易价格", "交易金额", "净亏损", "中标金额", "债务规模", "发行价格", "市值", "募资金额", "融资金额"],'时间':['披露时间',"事件时间", "回购完成时间", "交易完成时间", "财报周期", "中标日期", "披露日期", "破产时间", "被约谈时间", "收购完成时间"], \
                '职位':["高管职位", "变动后职位"], '中标标的':["中标标的"], '收购标的':["收购标的"], '融资轮次':["融资轮次"], '变动类型':["亏损变化", "变动类型"],'质押物':["质押物"]}
