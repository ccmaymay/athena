from setuptools import setup, Command
from setuptools.extension import Extension
from Cython.Build import cythonize
from glob import glob
import numpy as np
import os
import shutil
import platform


__version__ = '0.2.5b0'


class _Clean(Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        for d in ('build', 'dist', 'athena.egg-info', 'test/__pycache__'):
            if os.path.isdir(d):
                print 'removing %s' % d
                shutil.rmtree(d)
        for pyx in glob('athena/*.pyx'):
            stem = pyx[:-len('.pyx')]
            for gen in (stem + '.c', stem + '.cpp'):
                if os.path.isfile(gen):
                    print 'removing %s' % gen
                    os.remove(gen)


def have_cblas():
    print 'checking for cblas ...'

    from subprocess import check_call
    from tempfile import mkstemp
    import os

    (in_fd, in_path) = mkstemp(suffix='.c')
    try:
        os.close(in_fd)
        (out_fd, out_path) = mkstemp(suffix='.c')
        try:
            os.close(out_fd)

            with open(in_path, 'w') as in_f:
                in_f.write(r'''
#include <cblas.h>
int main(int argc, char **argv) {
  double x[] = {0, 1, 2};
  double y[] = {4, 7, 5};
  double z = cblas_ddot(3, x, 1, y, 1);
  return (z - 17 < 1e-6 && z - 17 > -1e-6) ? 0 : 1;
}
''')
            cblas_libs = ['cblas']
            base_args = ['gcc', '-O0', '-g', '-Wall', '-Werror', '-o',
                         out_path, in_path]

            try:
                args = base_args + map(lambda l: '-l%s' % l, cblas_libs)
                print ' '.join(args)
                check_call(args)
                check_call(out_path)
                print 'found cblas'
                return cblas_libs

            except:
                try:
                    cblas_libs += ['atlas']
                    args = base_args + map(lambda l: '-l%s' % l, cblas_libs)
                    print ' '.join(args)
                    check_call(args)
                    check_call(out_path)
                    print 'found cblas (atlas)'
                    return cblas_libs
                except:
                    pass

        finally:
            if os.path.isfile(out_path):
                os.remove(out_path)

    finally:
        if os.path.isfile(in_path):
            os.remove(in_path)

    print 'cblas not found'
    return None


extra_compile_args = [
    '-std=gnu++11', '-Wall', '-DLOG_INFO'
]
extra_link_args = []
libraries = []


if platform.system().lower() == 'darwin':
    extra_compile_args += ['-Wno-unused-function', '-Wno-unknown-pragmas']
    extra_link_args += ['-framework', 'Accelerate']
else:
    extra_compile_args += ['-fopenmp']
    extra_link_args += ['-fopenmp']
    cblas_libs = have_cblas()
    if cblas_libs:
        extra_compile_args += ['-DHAVE_CBLAS']
        libraries += cblas_libs


setup(
    name='athena',
    version=__version__,
    description='little framework of streaming algorithms',
    packages=[
        'athena',
    ],
    ext_modules=cythonize([
        Extension(
            'athena.core', ['athena/core.pyx',
                            'athena/_core.cpp', 'athena/_math.cpp',
                            'athena/_sgns.cpp', 'athena/_cblas.cpp',
                            'athena/_word2vec.cpp', 'athena/_io.cpp',
                            'athena/_log.cpp'],
            language='c++',
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            libraries=libraries,
        ),
    ]),
    include_dirs=[np.get_include(), 'athena'],
    scripts=glob('scripts/*'),
    cmdclass={'clean': _Clean},
    install_requires=[
        'numpy',
        'concrete',
        'redis>=2.10.0',
    ],
    url='https://gitlab.hltcoe.jhu.edu/littleowl/athena',
    license='BSD',
)
