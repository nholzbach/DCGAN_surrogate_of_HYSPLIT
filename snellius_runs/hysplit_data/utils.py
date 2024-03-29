"""
Utility functions for automate-plume-viz
This code was taken and edited from the following path on hal21 server (on Oct 26, 2020):
    /projects/earthtime/files/src/python-utils/utils.ipynb
"""


import os, requests, concurrent, concurrent.futures, datetime, math, shutil, subprocess, sys, time, traceback, urllib
from requests.exceptions import RequestException
from contextlib import closing

try:
    import dateutil, dateutil.tz
except:
    pass


def log(*args):
    global logfile
    try:
        logfile
    except:
        logfile = None
    if logfile:
        logfile.write('%s %d: %s\n' % (datetime.datetime.now().isoformat(' '),
                                       os.getpid(),
                                       ' '.join(args)))
        logfile.flush()
    sys.stderr.write('%s\n' % (' '.join(args)))


def start_logging(path):
    global logfile
    logfile = open(path, 'a')
    log('%s logging started' % __file__)


def subprocess_check(*args, **kwargs):
    verbose = kwargs.pop('verbose', False)
    ignore_error = kwargs.pop('ignore_error', False)
    if len(args) == 1 and type(args[0]) == str:
        kwargs['shell'] = True
        if verbose:
            print(args[0])
    elif verbose:
        print(' '.join(args[0]))
    p = subprocess.Popen(
        *args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs)
    (out, err) = p.communicate()
    out = out.decode('utf8')
    err = err.decode('utf8')
    ret = p.wait()
    if ret != 0 and not ignore_error:
        raise Exception(
            ('Call to subprocess_check failed with return code {ret}\n'
             'Standard error:\n{err}'
             'Standard out:\n{out}').format(**locals()))
    if len(err) > 0 and len(out) > 0 and err[-1] != '\n':
        err += '\n'
    all = err + out
    if verbose and all.strip():
        print(all.strip())
    return all

def download_file(url, filename, timeout=3600):
    if os.path.exists(filename):
        sys.stdout.write('%s already downloaded\n' % filename)
        return True
    else:
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        sys.stdout.write('Downloading %s to %s\n' % (url, filename))
        
        if url.startswith('ftp'):
            with closing(urllib.request.urlopen(url,timeout=timeout)) as r:
                with open(filename + '.tmp', 'wb') as f:
                    shutil.copyfileobj(r, f)
            sys.stdout.write('Done, wrote file to %s\n' % (filename))
        else:
        
            try:
                response = requests.Session().get(url, timeout=timeout)
                if(response.status_code!=200):
                    print('Error response, code = %d, body = %s' % (response.status_code, response.text))
                    return False
            except RequestException as e:
                sys.stdout.write("Couldn't read %s because %s" % (url, e))
                return False

            open(filename + '.tmp', "wb").write(response.content)
            sys.stdout.write('Done, wrote %d bytes to %s\n' % (len(response.content), filename))
        os.rename(filename + '.tmp', filename)
        return True


def unzip_file(filename):
    exdir = os.path.splitext(filename)[0]
    if os.path.exists(exdir):
        sys.stdout.write('%s already unzipped\n' % (filename))
    else:
        tmpdir = exdir + '.tmp'
        shutil.rmtree(tmpdir, True)
        sys.stdout.write('Unzipping %s into %s\n' % (filename, tmpdir))
        subprocess_check(['unzip', filename, '-d', tmpdir])
        os.rename(tmpdir, exdir)
        print('Success, created %s' % exdir)
    return exdir


def gunzip_file(filename):
    dest = os.path.splitext(filename)[0]
    if os.path.exists(dest):
        sys.stdout.write('%s already unzipped\n' % (filename))
    else:
        tmp = dest + '.tmp'
        sys.stdout.write('gunzipping %s\n' % (filename))
        subprocess.check_call("gunzip -c '%s' > '%s'" % (filename, tmp), shell=True)
        os.rename(tmp, dest)
        sys.stdout.write('Success, created %s\n' % (dest))


class SimpleThreadPoolExecutor(concurrent.futures.ThreadPoolExecutor):
    """Raises worker exceptions in shutdown"""
    def __init__(self, max_workers):
        super(SimpleThreadPoolExecutor, self).__init__(max_workers=max_workers)
        self.futures = []

    def submit(self, fn, *args, **kwargs):
        future = super(SimpleThreadPoolExecutor, self).submit(fn, *args, **kwargs)
        self.futures.append(future)
        return future

    def get_futures(self):
        return self.futures

    def shutdown(self):
        exception_count = 0
        results = []
        for completed in concurrent.futures.as_completed(self.futures):
            try:
                results.append(completed.result())
            except Exception:
                exception_count += 1
                sys.stderr.write(
                    'Exception caught in SimpleThreadPoolExecutor.shutdown.  Continuing until all are finished.\n' +
                    'Exception follows:\n' +
                    traceback.format_exc())
        super(SimpleThreadPoolExecutor, self).shutdown()
        if exception_count:
            raise Exception('SimpleThreadPoolExecutor failed: %d of %d raised exception' % (exception_count, len(self.futures)))
        print('SimpleThreadPoolExecutor succeeded: all %d jobs completed' % len(self.futures))
        return results


class SimpleProcessPoolExecutor(concurrent.futures.ProcessPoolExecutor):
    def __init__(self, max_workers):
        super(SimpleProcessPoolExecutor, self).__init__(max_workers=max_workers)
        self.futures = []

    def submit(self, fn, *args, **kwargs):
        future = super(SimpleProcessPoolExecutor, self).submit(fn, *args, **kwargs)
        self.futures.append(future)
        return future

    def get_futures(self):
        return self.futures

    def shutdown(self):
        exception_count = 0
        results = []
        for completed in concurrent.futures.as_completed(self.futures):
            try:
                results.append(completed.result())
            except Exception:
                exception_count += 1
                sys.stderr.write(
                    'Exception caught in SimpleProcessPoolExecutor.shutdown.  Continuing until all are finished.\n' +
                    'Exception follows:\n' +
                    traceback.format_exc())
        super(SimpleProcessPoolExecutor, self).shutdown()
        if exception_count:
            raise Exception('SimpleProcessPoolExecutor failed: %d of %d raised exception' % (exception_count, len(self.futures)))
        print('SimpleProcessPoolExecutor succeeded: all %d jobs completed' % len(self.futures))
        return results

    def kill(self, signal=9):
        for pid in self._processes.keys():
            print('Killing %d with signal %d' % (pid, signal))
            os.kill(pid, signal)


class Stopwatch:
    """
    Usage:
        with Stopwatch('Sleeping for half a second'):
            time.sleep(0.5)
    """
    def __init__(self, name):
        self.name = name
    def __enter__(self):
        self.start = time.time()
    def __exit__(self, type, value, traceback):
        sys.stdout.write('%s took %.1f seconds\n' % (self.name, time.time() - self.start))
        sys.stdout.flush()


def sleep_until_next_period(period, offset=0):
    now = time.time()
    start_of_next_period = math.ceil((now - offset) / period) * period + offset
    delay = start_of_next_period - now
    print('sleep_until_next_period(%d, %d) sleeping %d seconds until %s' %
          (period, offset, delay, datetime.datetime.fromtimestamp(start_of_next_period).strftime('%H:%M:%S')))
    time.sleep(delay)


def formatSecs(secs):
    if secs < 60:
        return '%d secs' % secs

    mins = secs / 60
    if mins < 60:
        return '%.1f mins' % mins

    hours = mins / 60;
    if hours < 24:
        return '%.1f hrs' % hours

    days = hours / 24
    return '%.1f days' % days


class StatInstance:
    """
    Usage:
        Stat = StatInstance()
        Stat.log('RAMP2ESDR', 'up', 'Upload succeeded and data is up-to-date')
    """
    def __init__(self, use_staging_server=False):
        if use_staging_server:
            self.server_hostname = 'stat-staging.createlab.org'
        else:
            self.server_hostname = 'stat.createlab.org'
        self.hostname = None
        self.service = None

    def get_datetime(self):
        return datetime.datetime.now(dateutil.tz.tzlocal()).isoformat()

    def get_hostname(self):
        if not self.hostname:
            self.hostname = subprocess_check('hostname').strip()
        return self.hostname

    def set_service(self, service):
        self.service = service

    # Possible levels include 'up', 'down', 'info', 'debug', 'warning', critical'
    def log(self, service, level, summary, details=None, host=None, payload={}, valid_for_secs=None, shortname=None):
        service = service or self.service
        if not service:
            raise Exception('log: service must be passed, or set previously with set_service')
        host = host or self.get_hostname()
        post_body = {
                'service': service,
                'datetime': self.get_datetime(),
                'host': host,
                'level': level,
                'summary': summary,
                'details': details,
                'payload': payload,
                'valid_for_secs': valid_for_secs,
                'shortname': shortname
            }
        print('Stat.log %s %s %s %s %s' % (level, service, host, summary, details))
        sys.stdout.flush()
        timeoutInSecs = 20
        try:
            response = requests.post('https://%s/api/log' % self.server_hostname,
                                     json=post_body, timeout=timeoutInSecs)
            if response.status_code != 200:
                sys.stderr.write('POST to https://stat.createlab.org/api/log failed with status code %d and response %s' % (response.status_code, response.text))
                sys.stderr.flush()
                return
        except RequestException:
            sys.stderr.write('POST to https://stat.createlab.org/api/log timed out')
            sys.stderr.flush()

    def info(self, summary, details=None, payload={}, host=None, service=None, shortname=None):
        self.log(service, 'info', summary, details=details, payload=payload, host=host, shortname=shortname)

    def debug(self, summary, details=None, payload={}, host=None, service=None, shortname=None):
        self.log(service, 'debug', summary, details=details, payload=payload, host=host, shortname=shortname)

    def warning(self, summary, details=None, payload={}, host=None, service=None, shortname=None):
        self.log(service, 'warning', summary, details=details, payload=payload, host=host, shortname=shortname)

    def critical(self, summary, details=None, payload={}, host=None, service=None, shortname=None):
        self.log(service, 'critical', summary, details=details, payload=payload, host=host, shortname=shortname)

    def up(self, summary, details=None, payload={}, valid_for_secs=None, host=None, service=None, shortname=None):
        self.log(service, 'up', summary,
                 details=details, payload=payload, valid_for_secs=valid_for_secs, host=host, shortname=shortname)

    def down(self, summary, details=None, payload={}, valid_for_secs=None, host=None, service=None, shortname=None):
        self.log(service, 'down', summary,
                 details=details, payload=payload, valid_for_secs=valid_for_secs, host=host, shortname=shortname)


def notebook_wide_display():
    # Wide display
    from IPython.core.display import display, HTML
    display(HTML("<style>#notebook-container { margin-left:-14px; width:calc(100% + 27px) !important; }</style>"))