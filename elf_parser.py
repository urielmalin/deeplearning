#!/usr/bin/python

import sys
from elftools.elf.elffile import ELFFile
from elftools.elf.sections import SymbolTableSection

class Function(object):
    def __init__(self, name, offset, size):
        self.name = name
        self.offset = offset
        self.size = size
    def __str__(self):
        return "%-30s\t0x%08x\t0x%08x" % (self.name , self.offset, self.size)

class FunctionsList(list):

    def __init__(self):
        list.__init__(self)

    def append(self, name, offset, size):
        list.append(self, Function(name, offset, size))

    def is_offset_start(self, offset):
        for func in self:
            if func.offset == offset:
                return True
        return False

    def get_offsets_list(self):
        return [func.offset for func in self]


class ElfParser(object):
    def __init__(self, f):
        self.elffile = ELFFile(open(f, "rb"))
        self.text_section = self.elffile.get_section_by_name(".text")
        self.code = self.text_section.data()
        self.code_len = len(self.code)
        self.text_offset = self.text_section.header.sh_addr
        self.funcs = FunctionsList() 
        self.init_functions_list()

    def get_functions_list(self):
        return self.funcs

    def get_code_and_funcs(self):
        return self.get_binary_code(), self.get_functions_list()

    def get_functions_num(self):
        return len(funcs)

    def get_code_len(self):
        return self.code_len

    def get_binary_code(self):
        return self.code

    def get_section_idx(self, section):
        for i in xrange(self.elffile.num_sections()):
            if self.elffile.get_section(i) == section:
                return i

    def va_to_offset(self, va):
        return va - self.text_offset

    def offset_to_va(self, offset):
        return offset + self.text_offset

    @staticmethod
    def is_function_symbol(symbol, section_idx):
            if symbol.entry.st_info.type == "STT_FUNC":
                if symbol.entry.st_shndx == section_idx:
                    if symbol.entry.st_size > 0:
                        return True
            return False

    def init_functions_list(self):
        symtab = self.elffile.get_section_by_name(".symtab")
        text_section_idx = self.get_section_idx(self.text_section)
        if not isinstance(symtab, SymbolTableSection):
            raise Exception
        for symbol in symtab.iter_symbols():
            if self.is_function_symbol(symbol, text_section_idx):
                sym_offset = self.va_to_offset(symbol.entry.st_value)
                self.funcs.append(symbol.name, sym_offset, symbol.entry.st_size)

    def print_functions_list(self):
        print "%-30s\t%8s\t%8s" % ("Name", "Offset", "Size")
        print "-" * 58
        for func in self.funcs:
            print func

        

def main(argv):
    ep = ElfParser(argv[1])
    ep.init_functions_list()
    ep.print_functions_list()

if __name__ == "__main__":
    main(sys.argv)
